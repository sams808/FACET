"""The OpenGL renderer.

Draws a :class:`~facet.gl.scene.Scene` through a G-buffer, then composites.
Runs on both GL tiers from one geometry path; the tier only chooses the shader
dialect and whether the ambient-occlusion pass runs.

PySide6 binding traps, all of which cost real time to find, so they are guarded
or commented at the point they apply:

* ``glVertexAttribPointer``'s offset must be a ``shiboken6.VoidPtr``. An ``int``
  and ``None`` are rejected, but **``ctypes.c_void_p`` is accepted and silently
  passes the wrong address** -- the draw then succeeds, raises no GL error, and
  renders nothing.
* its ``normalized`` argument is bound as an ``int``, not a ``bool``.
* ``glUniformMatrix4fv`` wants a flat ``Sequence[float]``, not ``bytes``.
* ``glGenVertexArrays`` and ``glGenFramebuffers`` are not exposed at all; use
  ``QOpenGLVertexArrayObject`` and ``QOpenGLFramebufferObject``.
* ``QOpenGLFramebufferObject.toImage()`` leaves the **default** framebuffer
  bound, so anything drawn after a readback goes to the wrong target.
"""
from __future__ import annotations

import numpy as np
import shiboken6
from PySide6.QtCore import QSize
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLFramebufferObject,
    QOpenGLFramebufferObjectFormat,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVertexArrayObject,
)

from . import buffers, shaders
from .caps import Capabilities, Tier
from .camera import Camera
from .scene import Scene

# GL constants, so the module does not depend on a particular enum wrapper
GL_FLOAT = 0x1406
GL_TRIANGLES = 0x0004
GL_LINES = 0x0001
GL_COLOR_BUFFER_BIT = 0x4000
GL_DEPTH_BUFFER_BIT = 0x0100
GL_DEPTH_TEST = 0x0B71
GL_BLEND = 0x0BE2
GL_CULL_FACE = 0x0B44
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_TEXTURE_2D = 0x0DE1
GL_TEXTURE0 = 0x84C0
GL_RGBA16F = 0x881A
GL_COLOR_ATTACHMENT0 = 0x8CE0
GL_LINE_SMOOTH = 0x0B20
GL_FRAMEBUFFER = 0x8D40


class ShaderError(RuntimeError):
    pass


class Renderer:
    """Owns GL resources for one context. Not reusable across contexts."""

    def __init__(self, gl, caps: Capabilities):
        self.gl = gl
        self.caps = caps
        self.version = caps.glsl_version
        self.use_ssao = caps.supports_ssao
        self.use_outline = caps.tier is Tier.FULL
        self.background = np.array([0.086, 0.094, 0.110], np.float32)
        self.cell_color = np.array([0.45, 0.48, 0.55], np.float32)
        self.fog_amount = 0.65
        self.supersample = 1
        # what the tier allows, kept separate from what the theme asks for, so
        # a theme requesting ambient occlusion on a software renderer is
        # honoured as far as it can be rather than silently ignored
        self._ssao_available = caps.supports_ssao
        self._outline_available = caps.tier is Tier.FULL

        self._programs: dict[str, QOpenGLShaderProgram] = {}
        self._vaos: dict[str, QOpenGLVertexArrayObject] = {}
        self._vbos: dict[str, dict[str, QOpenGLBuffer]] = {}
        self._counts: dict[str, int] = {}
        self._gbuffer = None
        self._ao = None
        self._composite = None
        self._size = QSize(0, 0)
        self._scene: Scene | None = None
        self._tube_sides = 16
        # When set, the composite pass draws straight into this framebuffer
        # instead of an owned one. A QOpenGLWidget renders into its own FBO
        # rather than the window, so this avoids an extra full-screen blit.
        self.target_fbo: int | None = None

        self._build_programs()
        self._build_quad()

    # -- theme -------------------------------------------------------------
    def apply_theme(self, theme) -> None:
        """Adopt a theme's presentation settings.

        Only the settings the tier can honour are taken. A theme asking for
        ambient occlusion on the software renderer keeps the request -- so that
        moving the same theme to a machine with a GPU turns it on -- but does
        not enable a pass that would make interaction unusable here.
        """
        self.background = np.array(theme.background, np.float32)
        self.cell_color = np.array(theme.cell_color, np.float32)
        self.fog_amount = float(theme.fog_amount)
        self.use_ssao = bool(theme.ambient_occlusion) and self._ssao_available
        self.use_outline = bool(theme.outlines) and self._outline_available

    # -- setup -------------------------------------------------------------
    def _build_programs(self) -> None:
        spec = [
            ("sphere", shaders.SPHERE_VS, shaders.SPHERE_FS, 3),
            ("tube", shaders.TUBE_VS, shaders.TUBE_FS, 3),
            ("poly", shaders.POLY_VS, shaders.POLY_FS, 1),
            ("line", shaders.LINE_VS, shaders.LINE_FS, 1),
            ("composite", shaders.QUAD_VS,
             shaders.COMPOSITE_FS if self.use_ssao else shaders.COMPOSITE_SIMPLE_FS, 1),
        ]
        if self.use_ssao:
            spec.append(("ssao", shaders.QUAD_VS, shaders.SSAO_FS, 1))
        for name, vs, fs, n_out in spec:
            self._programs[name] = self._compile(name, vs, fs, n_out)

    def _compile(self, name: str, vs: str, fs: str,
                 n_outputs: int) -> QOpenGLShaderProgram:
        p = QOpenGLShaderProgram()
        if not p.addShaderFromSourceCode(QOpenGLShader.Vertex,
                                         shaders.emit(vs, self.version)):
            raise ShaderError(f"{name} vertex shader:\n{p.log()}")
        if not p.addShaderFromSourceCode(QOpenGLShader.Fragment,
                                         shaders.emit(fs, self.version)):
            raise ShaderError(f"{name} fragment shader:\n{p.log()}")

        # GLSL 130 has no layout qualifiers, so both attribute and fragment
        # output locations are bound by name before linking. Doing it on every
        # tier keeps the two paths identical rather than merely equivalent.
        for attr, loc in shaders.ATTRIBUTES.items():
            p.bindAttributeLocation(attr, loc)
        if self.version < 330:
            for i, out in enumerate(shaders.FRAG_OUTPUTS[:n_outputs]):
                try:
                    self.gl.glBindFragDataLocation(p.programId(), i, out)
                except Exception:
                    pass

        if not p.link():
            raise ShaderError(f"{name} link:\n{p.log()}")
        return p

    def _build_quad(self) -> None:
        corners = np.array([[-1, -1], [1, -1], [1, 1],
                            [-1, -1], [1, 1], [-1, 1]], np.float32)
        self._make_vao("quad", {"aCorner": corners}, len(corners))

    # -- buffers -----------------------------------------------------------
    def _make_vao(self, key: str, arrays: dict[str, np.ndarray], count: int) -> None:
        gl = self.gl
        vao = self._vaos.get(key)
        if vao is None:
            vao = QOpenGLVertexArrayObject()
            vao.create()
            self._vaos[key] = vao
            self._vbos[key] = {}
        vao.bind()
        for name, data in arrays.items():
            loc = shaders.ATTRIBUTES[name]
            arr = np.ascontiguousarray(data, np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            buf = self._vbos[key].get(name)
            if buf is None:
                buf = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
                buf.create()
                self._vbos[key][name] = buf
            buf.bind()
            buf.allocate(arr.tobytes(), arr.nbytes)
            gl.glEnableVertexAttribArray(loc)
            # normalized is an int here, and the offset MUST be a VoidPtr --
            # ctypes.c_void_p is accepted and silently wrong.
            gl.glVertexAttribPointer(loc, arr.shape[1], GL_FLOAT, 0, 0,
                                     shiboken6.VoidPtr(0))
            buf.release()
        vao.release()
        self._counts[key] = count

    def set_scene(self, scene: Scene) -> None:
        """Upload a scene. Call on load, on a style change, or after restyling."""
        self._scene = scene
        self._tube_sides = buffers.tube_sides_for(
            scene, budget=400_000 if self.caps.tier is Tier.FULL else 120_000)

        sph = buffers.sphere_vertices(scene)
        self._make_vao("sphere", sph, len(sph["aCorner"]))

        tube = buffers.tube_vertices(scene, self._tube_sides)
        self._make_vao("tube", tube, len(tube["aPosition"]))

        if scene.n_poly_triangles:
            self._make_vao("poly", {"aPosition": scene.poly_vertices,
                                    "aNormal": scene.poly_normals},
                           len(scene.poly_vertices))
        else:
            self._counts["poly"] = 0

        lines = buffers.line_vertices(scene.cell_segments)
        if len(lines):
            self._make_vao("line", {"aPosition": lines}, len(lines))
        else:
            self._counts["line"] = 0

    def update_bond_colors(self, scene: Scene) -> None:
        """Re-upload only what a threshold change touches.

        Dragging the cutoff changes bond radii and colours but not the topology,
        so the sphere, polyhedron and cell buffers are left alone.
        """
        if scene.n_bonds == 0:
            return
        tube = buffers.tube_vertices(scene, self._tube_sides)
        self._make_vao("tube", tube, len(tube["aPosition"]))

    # -- framebuffers ------------------------------------------------------
    def resize(self, width: int, height: int) -> None:
        w = max(1, int(width) * self.supersample)
        h = max(1, int(height) * self.supersample)
        if self._size == QSize(w, h) and self._gbuffer is not None:
            return
        self._size = QSize(w, h)

        fmt = QOpenGLFramebufferObjectFormat()
        fmt.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
        self._gbuffer = QOpenGLFramebufferObject(self._size, fmt)
        self._gbuffer.addColorAttachment(self._size, GL_RGBA16F)   # view position
        self._gbuffer.addColorAttachment(self._size, GL_RGBA16F)   # normal

        plain = QOpenGLFramebufferObjectFormat()
        self._composite = QOpenGLFramebufferObject(self._size, plain)
        self._ao = QOpenGLFramebufferObject(self._size, plain) if self.use_ssao else None

    # -- drawing -----------------------------------------------------------
    def render(self, camera: Camera, width: int, height: int,
               picking: bool = False) -> QOpenGLFramebufferObject:
        """Draw the scene and return the framebuffer holding the result."""
        gl = self.gl
        self.resize(width, height)
        w, h = self._size.width(), self._size.height()
        scene = self._scene

        view, proj = camera.matrices(w / max(h, 1))
        light = np.array([0.42, 0.62, 0.66])
        light = light / np.linalg.norm(light)
        light_view = view[:3, :3] @ light

        self._gbuffer.bind()
        self._draw_buffers(3)
        gl.glViewport(0, 0, w, h)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        gl.glEnable(GL_DEPTH_TEST)
        gl.glDisable(GL_BLEND)

        if scene is not None:
            self._draw_spheres(view, proj, light_view, picking)
            self._draw_tubes(view, proj, light_view, picking)
            if not picking:
                self._draw_lines(view, proj)
                self._draw_polyhedra(view, proj, light_view, scene)

        self._gbuffer.release()

        if picking:
            return self._gbuffer

        if self.use_ssao:
            self._pass_ssao(proj, camera)
        self._pass_composite(camera)
        return self._composite

    def _draw_buffers(self, n: int) -> None:
        try:
            self.gl.glDrawBuffers(n, [GL_COLOR_ATTACHMENT0 + i for i in range(n)])
        except Exception:
            pass

    def _draw_spheres(self, view, proj, light, picking) -> None:
        if not self._counts.get("sphere"):
            return
        p = self._programs["sphere"]
        p.bind()
        self._set_matrix(p, "uView", view)
        self._set_matrix(p, "uProj", proj)
        self.gl.glUniform3f(p.uniformLocation("uLight"), *light)
        self.gl.glUniform1i(p.uniformLocation("uPicking"), 1 if picking else 0)
        self._vaos["sphere"].bind()
        self.gl.glDrawArrays(GL_TRIANGLES, 0, self._counts["sphere"])
        self._vaos["sphere"].release()
        p.release()

    def _draw_tubes(self, view, proj, light, picking) -> None:
        if not self._counts.get("tube"):
            return
        p = self._programs["tube"]
        p.bind()
        self._set_matrix(p, "uView", view)
        self._set_matrix(p, "uProj", proj)
        self.gl.glUniform3f(p.uniformLocation("uLight"), *light)
        self.gl.glUniform1i(p.uniformLocation("uPicking"), 1 if picking else 0)
        # bonds are not pickable targets yet; they write id zero so a click that
        # lands on one reads as background rather than as a wrong atom
        self.gl.glUniform3f(p.uniformLocation("uPickId"), 0.0, 0.0, 0.0)
        self._vaos["tube"].bind()
        self.gl.glDrawArrays(GL_TRIANGLES, 0, self._counts["tube"])
        self._vaos["tube"].release()
        p.release()

    def _draw_lines(self, view, proj) -> None:
        if not self._counts.get("line"):
            return
        p = self._programs["line"]
        p.bind()
        self._set_matrix(p, "uView", view)
        self._set_matrix(p, "uProj", proj)
        self.gl.glUniform3f(p.uniformLocation("uColor"), *self.cell_color)
        self.gl.glUniform1f(p.uniformLocation("uAlpha"), 0.85)
        self._vaos["line"].bind()
        self.gl.glDrawArrays(GL_LINES, 0, self._counts["line"])
        self._vaos["line"].release()
        p.release()

    def _draw_polyhedra(self, view, proj, light, scene: Scene) -> None:
        if not self._counts.get("poly"):
            return
        gl = self.gl
        # colour attachment only: a transparent surface must not overwrite the
        # view position and normal the ambient-occlusion pass reads
        self._draw_buffers(1)
        gl.glEnable(GL_BLEND)
        gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        gl.glDepthMask(False)
        gl.glDisable(GL_CULL_FACE)

        p = self._programs["poly"]
        p.bind()
        self._set_matrix(p, "uView", view)
        self._set_matrix(p, "uProj", proj)
        gl.glUniform3f(p.uniformLocation("uLight"), *light)
        gl.glUniform3f(p.uniformLocation("uColor"), *scene.poly_color)
        gl.glUniform1f(p.uniformLocation("uAlpha"), scene.poly_alpha)
        self._vaos["poly"].bind()
        gl.glDrawArrays(GL_TRIANGLES, 0, self._counts["poly"])
        self._vaos["poly"].release()
        p.release()

        gl.glDepthMask(True)
        gl.glDisable(GL_BLEND)
        self._draw_buffers(3)

    # -- full-screen passes ------------------------------------------------
    def _pass_ssao(self, proj, camera: Camera) -> None:
        gl = self.gl
        tex = self._gbuffer.textures()
        self._ao.bind()
        gl.glViewport(0, 0, self._size.width(), self._size.height())
        gl.glDisable(GL_DEPTH_TEST)
        gl.glClear(GL_COLOR_BUFFER_BIT)
        p = self._programs["ssao"]
        p.bind()
        self._set_matrix(p, "uProj", proj)
        gl.glUniform2f(p.uniformLocation("uRes"),
                       self._size.width(), self._size.height())
        gl.glUniform1f(p.uniformLocation("uRadius"),
                       max(1.1, camera.scene_radius * 0.20))
        self._bind_texture(p, "uPos", tex[1], 0)
        self._bind_texture(p, "uNormal", tex[2], 1)
        self._draw_quad()
        p.release()
        self._ao.release()

    def _pass_composite(self, camera: Camera) -> None:
        gl = self.gl
        tex = self._gbuffer.textures()
        near, far = camera.clip_planes()
        if self.target_fbo is None:
            self._composite.bind()
        else:
            gl.glBindFramebuffer(GL_FRAMEBUFFER, int(self.target_fbo))
        gl.glViewport(0, 0, self._size.width(), self._size.height())
        gl.glDisable(GL_DEPTH_TEST)
        gl.glClear(GL_COLOR_BUFFER_BIT)

        p = self._programs["composite"]
        p.bind()
        gl.glUniform3f(p.uniformLocation("uBg"), *self.background)
        gl.glUniform1f(p.uniformLocation("uFogNear"),
                       camera.distance - camera.scene_radius * 0.55)
        gl.glUniform1f(p.uniformLocation("uFogFar"),
                       camera.distance + camera.scene_radius * 1.25)
        gl.glUniform1f(p.uniformLocation("uFogAmount"), self.fog_amount)
        self._bind_texture(p, "uColor", tex[0], 0)
        if self.use_ssao:
            gl.glUniform2f(p.uniformLocation("uRes"),
                           self._size.width(), self._size.height())
            gl.glUniform1i(p.uniformLocation("uUseAO"), 1)
            gl.glUniform1i(p.uniformLocation("uOutline"), 1 if self.use_outline else 0)
            self._bind_texture(p, "uAO", self._ao.texture(), 1)
            self._bind_texture(p, "uPos", tex[1], 2)
        else:
            self._bind_texture(p, "uPos", tex[1], 1)
        self._draw_quad()
        p.release()
        if self.target_fbo is None:
            self._composite.release()

    def reset_state(self) -> None:
        """Hand the context back in a state QPainter can draw into.

        QPainter has its own OpenGL paint engine, and it assumes a clean
        context. Left as the geometry pass leaves it -- depth test on, a custom
        program bound, a vertex array bound -- every glyph QPainter draws fails
        the depth test against a buffer already full of near values. Nothing is
        drawn, no GL error is raised, and the overlay simply never appears.
        """
        gl = self.gl
        # Note: QOpenGLFramebufferObject.bindDefault() is NOT wanted here. It
        # binds framebuffer 0, and a QOpenGLWidget's default framebuffer is its
        # own rather than 0, so calling it would send the overlay to a surface
        # that is never shown. The overlay uses a QOpenGLPaintDevice instead,
        # which draws into whatever is currently bound.
        try:
            gl.glDisable(GL_DEPTH_TEST)
            gl.glDisable(GL_BLEND)
            gl.glDisable(GL_CULL_FACE)
            gl.glDepthMask(True)
            gl.glUseProgram(0)
            gl.glBindVertexArray(0)
            gl.glActiveTexture(GL_TEXTURE0)
            gl.glBindTexture(GL_TEXTURE_2D, 0)
        except Exception:
            # a tier that cannot do one of these is no worse off than before
            pass

    def _draw_quad(self) -> None:
        self._vaos["quad"].bind()
        self.gl.glDrawArrays(GL_TRIANGLES, 0, self._counts["quad"])
        self._vaos["quad"].release()

    def _bind_texture(self, program, uniform: str, texture: int, unit: int) -> None:
        self.gl.glActiveTexture(GL_TEXTURE0 + unit)
        self.gl.glBindTexture(GL_TEXTURE_2D, texture)
        self.gl.glUniform1i(program.uniformLocation(uniform), unit)

    def _set_matrix(self, program, name: str, m: np.ndarray) -> None:
        # a flat Sequence[float], not bytes; transpose=1 because our matrices
        # are row-major acting on column vectors and GLSL expects column-major
        self.gl.glUniformMatrix4fv(program.uniformLocation(name), 1, 1,
                                   np.asarray(m, np.float32).ravel().tolist())

    # -- picking -----------------------------------------------------------
    def pick(self, camera: Camera, width: int, height: int,
             x: int, y: int) -> int | None:
        """The atom index under a viewport pixel, or None.

        Reads an id written by the geometry itself rather than intersecting a
        ray on the CPU, so what is picked is exactly what is drawn -- including
        the analytic sphere silhouettes, which no CPU ray test would match.
        """
        self.render(camera, width, height, picking=True)
        self._gbuffer.bind()
        try:
            px = int(x) * self.supersample
            py = (int(height) - 1 - int(y)) * self.supersample   # GL origin is bottom-left
            if not (0 <= px < self._size.width() and 0 <= py < self._size.height()):
                return None
            image = self._gbuffer.toImage()
        finally:
            # toImage() leaves the DEFAULT framebuffer bound; release explicitly
            # so the next draw does not go somewhere unexpected
            self._gbuffer.release()
        if image.isNull():
            return None
        px = min(max(px, 0), image.width() - 1)
        py = min(max(py, 0), image.height() - 1)
        c = image.pixelColor(px, image.height() - 1 - py)
        return buffers.rgb_to_id(c.red(), c.green(), c.blue())

    # -- output ------------------------------------------------------------
    def to_image(self, camera: Camera, width: int, height: int,
                 supersample: int = 1):
        """Render once and return a QImage, optionally supersampled.

        Supersampling rather than multisampling: this driver refuses MSAA on the
        default framebuffer, and a downsampled 3x render is what a 600 dpi
        figure wants anyway.
        """
        from PySide6.QtCore import Qt

        previous = self.supersample
        self.supersample = max(1, int(supersample))
        try:
            self.resize(width, height)
            fbo = self.render(camera, width, height)
            image = fbo.toImage()
        finally:
            self.supersample = previous
            self._size = QSize(0, 0)          # force a rebuild at the old size
        if self.supersample != 1 or supersample != 1:
            image = image.scaled(width, height, Qt.KeepAspectRatio,
                                 Qt.SmoothTransformation)
        return image

    def release(self) -> None:
        for vao in self._vaos.values():
            vao.destroy()
        for group in self._vbos.values():
            for buf in group.values():
                buf.destroy()
        self._vaos.clear()
        self._vbos.clear()
        self._programs.clear()
        self._gbuffer = self._ao = self._composite = None
