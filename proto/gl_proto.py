"""Offscreen proof-of-concept renderer for the CIF explorer.

Purpose: demonstrate that publication-grade interactive 3D is reachable with
PySide6 alone -- no VTK, no pyvista, no PyOpenGL, no QWebEngine.

Technique stack (this is what makes molecular graphics look expensive):
  * ray-traced sphere impostors -- one instanced quad per atom, the sphere is
    intersected analytically in the fragment shader and gl_FragDepth is written.
    Mathematically perfect silhouettes at every zoom, no tessellation, and the
    atom count ceiling is set by fill rate rather than triangle count.
  * instanced tube meshes for bonds, split at the midpoint so each half carries
    its own atom colour.
  * a G-buffer (lit colour + view position + normal) feeding screen-space
    ambient occlusion, which is the single biggest contributor to the
    "solid object" impression and is what VESTA lacks.
  * depth cueing towards the background, CrystalMaker's signature look.
  * a silhouette pass driven by view-position discontinuity.
  * 3x supersampling, downsampled at the end. Cheaper to reason about than
    MSAA on the default framebuffer, which this driver declines to grant, and
    it is also exactly what a 600 dpi figure export wants.

Run:  py -3.11 gl_proto.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import shiboken6
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QGuiApplication, QOffscreenSurface, QOpenGLContext, QSurfaceFormat
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLFramebufferObject,
    QOpenGLFramebufferObjectFormat,
    QOpenGLFunctions_3_3_Core,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVertexArrayObject,
)

DEBUG = True
GL_RGBA16F = 0x881A
GL_COLOR_ATTACHMENT0 = 0x8CE0
GL_COLOR_ATTACHMENT1 = 0x8CE1
GL_COLOR_ATTACHMENT2 = 0x8CE2

# ---------------------------------------------------------------------------
# structure input
# ---------------------------------------------------------------------------

# Bond-valence parameters, Brese & O'Keeffe 1991 Table 2, b = 0.37 A.
# The real application ships the full IUCr accumulated table; a prototype only
# needs enough pairs to draw one structure.
BV = {("Bi", "O"): 2.09, ("Bi", "F"): 1.99, ("Si", "O"): 1.624,
      ("P", "O"): 1.617, ("Na", "O"): 1.803, ("B", "O"): 1.371,
      ("V", "O"): 1.803, ("Ba", "O"): 2.285, ("Si", "F"): 1.58,
      ("Na", "F"): 1.677, ("Mo", "O"): 1.907, ("W", "O"): 1.917,
      ("Al", "O"): 1.651, ("Fe", "O"): 1.759, ("Ti", "O"): 1.815}
B_BV = 0.37
V_BOND = 0.075          # v.u. -- a contact is a bond above this
V_LIST = 0.02           # v.u. -- a contact is worth tabulating above this

# VESTA-flavoured palette; deliberately less saturated than raw CPK so that
# ambient occlusion and depth cueing stay visible on top of it.
COLORS = {
    "Mo": (0.36, 0.66, 0.66), "W": (0.30, 0.52, 0.66), "Al": (0.74, 0.62, 0.55),
    "Fe": (0.87, 0.46, 0.22), "Ti": (0.62, 0.66, 0.70),
    "Bi": (0.62, 0.31, 0.71), "O": (0.94, 0.30, 0.24), "Na": (0.67, 0.80, 0.36),
    "Si": (0.42, 0.55, 0.78), "P": (0.95, 0.60, 0.22), "B": (0.55, 0.85, 0.66),
    "F": (0.55, 0.88, 0.80), "V": (0.85, 0.42, 0.42), "Ba": (0.34, 0.76, 0.55),
}
RADII = {"Mo": 0.44, "W": 0.44, "Al": 0.36, "Fe": 0.40, "Ti": 0.42, "Bi": 0.62, "O": 0.32, "Na": 0.52, "Si": 0.38, "P": 0.36,
         "B": 0.28, "F": 0.30, "V": 0.42, "Ba": 0.66}


def load_structure(path: str, pad: float = 4.0):
    """Expand a CIF to a cluster: every atom of the cell plus enough periodic
    images to close the coordination of the atoms near the cell boundary."""
    import gemmi

    doc = gemmi.cif.read(path)
    st = gemmi.make_small_structure_from_block(doc.sole_block())
    cell = st.cell
    M = np.array(cell.orth.mat.tolist(), float)  # fractional -> cartesian

    base = []
    for s in st.get_all_unit_cell_sites():
        el = s.element.name
        f = np.array([s.fract.x, s.fract.y, s.fract.z]) % 1.0
        base.append((el, f, float(s.occ), s.label))

    # how many periodic images are needed to reach `pad` angstrom beyond the cell
    reps = [max(1, int(math.ceil(pad / L)) ) for L in (cell.a, cell.b, cell.c)]
    atoms = []
    for el, f, occ, label in base:
        for i in range(-reps[0], reps[0] + 1):
            for j in range(-reps[1], reps[1] + 1):
                for k in range(-reps[2], reps[2] + 1):
                    ff = f + np.array([i, j, k], float)
                    xyz = M @ ff
                    atoms.append((el, xyz, occ, label, (i, j, k)))

    core = [a for a in atoms if a[4] == (0, 0, 0)]
    return st, cell, M, atoms, core


def find_bonds(atoms, core, vmin=V_BOND):
    """Contacts by partial bond valence, not by distance. v = exp((R0-d)/b).
    vmin = V_BOND gives the bonds; vmin = V_LIST gives everything worth
    tabulating, i.e. including the contacts whose status is the whole argument."""
    pos = np.array([a[1] for a in atoms])
    els = [a[0] for a in atoms]
    core_idx = [i for i, a in enumerate(atoms) if a[4] == (0, 0, 0)]
    bonds = []
    for i in core_idx:
        d = np.linalg.norm(pos - pos[i], axis=1)
        for j in np.where((d > 1.2) & (d < 4.5))[0]:
            if j <= i and j in core_idx:
                continue
            pair = (els[i], els[j])
            r0 = BV.get(pair) or BV.get((els[j], els[i]))
            if r0 is None:
                continue
            v = math.exp((r0 - d[j]) / B_BV)
            if v > vmin:
                bonds.append((i, int(j), v))
    return bonds


# ---------------------------------------------------------------------------
# shaders
# ---------------------------------------------------------------------------

SPHERE_VS = """
#version 330 core
layout(location=0) in vec2 quad;        // unit quad, -1..1
layout(location=1) in vec3 iCenter;     // world space
layout(location=2) in float iRadius;
layout(location=3) in vec3 iColor;
uniform mat4 uView;
uniform mat4 uProj;
out vec3 vCenterView;
out float vRadius;
out vec3 vColor;
out vec3 vRayView;
void main(){
    vec4 cv = uView * vec4(iCenter, 1.0);
    vCenterView = cv.xyz;
    vRadius = iRadius;
    vColor = iColor;
    // billboard slightly oversized so the silhouette is never clipped
    vec3 p = cv.xyz + vec3(quad * iRadius * 1.15, 0.0);
    vRayView = p;
    gl_Position = uProj * vec4(p, 1.0);
}
"""

SPHERE_FS = """
#version 330 core
in vec3 vCenterView;
in float vRadius;
in vec3 vColor;
in vec3 vRayView;
uniform mat4 uProj;
uniform vec3 uLight;      // view-space light direction
layout(location=0) out vec4 oColor;
layout(location=1) out vec4 oPos;
layout(location=2) out vec4 oNormal;
void main(){
    // ray from the eye through this fragment, intersected with the sphere
    vec3 rd = normalize(vRayView);
    vec3 oc = -vCenterView;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - vRadius * vRadius;
    float h = b * b - c;
    if (h < 0.0) discard;
    float t = -b - sqrt(h);
    vec3 p = rd * t;                       // view-space hit point
    vec3 n = normalize(p - vCenterView);

    vec4 clip = uProj * vec4(p, 1.0);
    gl_FragDepth = 0.5 * (clip.z / clip.w) + 0.5;

    float diff = max(dot(n, uLight), 0.0);
    vec3 h2 = normalize(uLight + vec3(0.0, 0.0, 1.0));
    float spec = pow(max(dot(n, h2), 0.0), 48.0);
    float rim = pow(1.0 - max(dot(n, vec3(0,0,1)), 0.0), 2.4);
    vec3 col = vColor * (0.26 + 0.74 * diff) + vec3(0.30) * spec + vColor * 0.16 * rim;

    oColor  = vec4(col, 1.0);
    oPos    = vec4(p, 1.0);
    oNormal = vec4(n, 1.0);
}
"""

TUBE_VS = """
#version 330 core
layout(location=0) in vec3 vert;        // unit tube: radius 1, length 1 along +z
layout(location=1) in vec3 nrm;
layout(location=2) in vec3 iA;          // bond start (world)
layout(location=3) in vec3 iB;          // bond end   (world)
layout(location=4) in float iRadius;
layout(location=5) in vec3 iColA;
layout(location=6) in vec3 iColB;
uniform mat4 uView;
uniform mat4 uProj;
out vec3 vPosView;
out vec3 vNormView;
out vec3 vColor;
void main(){
    vec3 ax = iB - iA;
    float len = length(ax);
    vec3 z = ax / len;
    vec3 t = abs(z.z) < 0.9 ? vec3(0,0,1) : vec3(1,0,0);
    vec3 x = normalize(cross(t, z));
    vec3 y = cross(z, x);
    mat3 basis = mat3(x, y, z);
    vec3 world = iA + basis * vec3(vert.xy * iRadius, vert.z * len);
    vec4 pv = uView * vec4(world, 1.0);
    vPosView  = pv.xyz;
    vNormView = normalize(mat3(uView) * (basis * nrm));
    vColor = vert.z < 0.5 ? iColA : iColB;
    gl_Position = uProj * pv;
}
"""

TUBE_FS = """
#version 330 core
in vec3 vPosView;
in vec3 vNormView;
in vec3 vColor;
uniform vec3 uLight;
layout(location=0) out vec4 oColor;
layout(location=1) out vec4 oPos;
layout(location=2) out vec4 oNormal;
void main(){
    vec3 n = normalize(vNormView);
    float diff = max(dot(n, uLight), 0.0);
    vec3 h = normalize(uLight + vec3(0,0,1));
    float spec = pow(max(dot(n, h), 0.0), 40.0);
    vec3 col = vColor * (0.30 + 0.70 * diff) + vec3(0.22) * spec;
    oColor  = vec4(col, 1.0);
    oPos    = vec4(vPosView, 1.0);
    oNormal = vec4(n, 1.0);
}
"""

# transparent coordination polyhedron, forward-rendered after the opaque pass
POLY_VS = """
#version 330 core
layout(location=0) in vec3 pos;
layout(location=1) in vec3 nrm;
uniform mat4 uView;
uniform mat4 uProj;
out vec3 vN;
out vec3 vP;
void main(){
    vec4 pv = uView * vec4(pos, 1.0);
    vP = pv.xyz;
    vN = normalize(mat3(uView) * nrm);
    gl_Position = uProj * pv;
}
"""

POLY_FS = """
#version 330 core
in vec3 vN;
in vec3 vP;
uniform vec3 uColor;
uniform float uAlpha;
uniform vec3 uLight;
layout(location=0) out vec4 oColor;
void main(){
    vec3 n = normalize(vN);
    if (!gl_FrontFacing) n = -n;
    float diff = max(dot(n, uLight), 0.0);
    // fresnel so the rim of the polyhedron reads even where it is nearly edge-on
    float fres = pow(1.0 - abs(dot(n, vec3(0,0,1))), 2.0);
    vec3 col = uColor * (0.45 + 0.55 * diff) + vec3(0.25) * fres;
    oColor = vec4(col, clamp(uAlpha + 0.42 * fres, 0.0, 0.92));
}
"""

FS_QUAD_VS = """
#version 330 core
layout(location=0) in vec2 quad;
out vec2 uv;
void main(){ uv = quad * 0.5 + 0.5; gl_Position = vec4(quad, 0.0, 1.0); }
"""

SSAO_FS = """
#version 330 core
in vec2 uv;
uniform sampler2D uPos;
uniform sampler2D uNormal;
uniform mat4 uProj;
uniform vec2 uRes;
uniform float uRadius;
out vec4 oAO;

const int NS = 24;
// fixed hemisphere kernel; a real implementation rotates it per-pixel by a
// tiled noise texture, which trades banding for high-frequency noise.
vec3 kern(int i){
    float a = float(i) * 2.39996323;          // golden angle
    float r = sqrt((float(i) + 0.5) / float(NS));
    // keep the hemisphere but lean the samples towards the tangent plane:
    // the occlusion signal on a ball-and-stick scene lives in the grazing
    // directions, where a neighbouring atom can actually block the sky.
    float z = 0.10 + 0.50 * r;
    return normalize(vec3(cos(a) * r, sin(a) * r, z)) * (0.25 + 0.75 * r);
}

void main(){
    vec4 P = texture(uPos, uv);
    if (P.w < 0.5) { oAO = vec4(1.0); return; }
    vec3 p = P.xyz;
    vec3 n = normalize(texture(uNormal, uv).xyz);
    vec3 t = abs(n.z) < 0.9 ? vec3(0,0,1) : vec3(1,0,0);
    vec3 bx = normalize(cross(t, n));
    vec3 by = cross(n, bx);
    mat3 TBN = mat3(bx, by, n);

    float occ = 0.0;
    for (int i = 0; i < NS; ++i){
        vec3 sp = p + TBN * kern(i) * uRadius;
        vec4 cl = uProj * vec4(sp, 1.0);
        vec2 suv = (cl.xy / cl.w) * 0.5 + 0.5;
        if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) continue;
        vec4 SP = texture(uPos, suv);
        if (SP.w < 0.5) continue;
        float sampleZ = SP.z;            // view space: less negative == nearer
        float rangeCheck = smoothstep(0.0, 1.0, uRadius / abs(p.z - sampleZ));
        if (sampleZ >= sp.z + 0.015) occ += rangeCheck;
    }
    oAO = vec4(vec3(clamp(1.0 - occ / float(NS), 0.0, 1.0)), 1.0);
}
"""

COMPOSITE_FS = """
#version 330 core
in vec2 uv;
uniform sampler2D uColor;
uniform sampler2D uAO;
uniform sampler2D uPos;
uniform vec2 uRes;
uniform float uFogNear;
uniform float uFogFar;
uniform vec3 uBg;
out vec4 oCol;

void main(){
    vec4 P = texture(uPos, uv);
    vec3 c = texture(uColor, uv).rgb;

    // blur the AO with a small box to kill kernel banding
    float ao = 0.0; float wsum = 0.0;
    for (int dy = -2; dy <= 2; ++dy)
      for (int dx = -2; dx <= 2; ++dx){
        vec2 o = vec2(dx, dy) / uRes;
        ao += texture(uAO, uv + o).r; wsum += 1.0;
      }
    ao /= wsum;

    if (P.w < 0.5) { oCol = vec4(uBg, 1.0); return; }

    // silhouette from view-position discontinuity
    float edge = 0.0;
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx){
        vec4 Q = texture(uPos, uv + vec2(dx, dy) * 1.6 / uRes);
        edge = max(edge, abs(Q.z - P.z) * (Q.w > 0.5 ? 1.0 : 0.0));
      }
    float outline = 1.0 - smoothstep(0.03, 0.16, edge);

    // ao comes back close to 1 over most of a convex scene; a power curve
    // expands the crevices into a range the eye can actually see.
    float aoc = pow(clamp(ao, 0.0, 1.0), 3.2);
    c *= mix(0.18, 1.0, aoc);          // ambient occlusion
    c *= mix(0.45, 1.0, outline);      // dark contact line

    // depth cue towards the background
    float f = clamp((-P.z - uFogNear) / (uFogFar - uFogNear), 0.0, 1.0);
    c = mix(c, uBg, f * 0.72);

    c = pow(c, vec3(1.0 / 1.9));       // gamma
    oCol = vec4(c, 1.0);
}
"""


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def unit_tube(sides: int = 20):
    """Unit tube of radius 1 running 0..1 along +z, as a triangle list with
    per-vertex normals. Rebased per instance in the vertex shader."""
    verts, norms = [], []
    for i in range(sides):
        a0 = 2 * math.pi * i / sides
        a1 = 2 * math.pi * (i + 1) / sides
        p0, p1 = (math.cos(a0), math.sin(a0)), (math.cos(a1), math.sin(a1))
        quad = [(p0, 0.0), (p1, 0.0), (p1, 1.0), (p0, 0.0), (p1, 1.0), (p0, 1.0)]
        for (xy, z) in quad:
            verts.append((xy[0], xy[1], z))
            norms.append((xy[0], xy[1], 0.0))
    # split the tube at the midpoint so each half can take its own colour
    out_v, out_n = [], []
    for (x, y, z), n in zip(verts, norms):
        out_v.append((x, y, z)); out_n.append(n)
    return np.array(out_v, np.float32), np.array(out_n, np.float32)


def split_tube(sides: int = 20):
    """Same tube but cut into two halves (0..0.5, 0.5..1) so the fragment
    shader's `vert.z < 0.5` colour test lands on a real vertex boundary."""
    verts, norms = [], []
    for half in (0, 1):
        z0, z1 = (0.0, 0.5) if half == 0 else (0.5, 1.0)
        for i in range(sides):
            a0 = 2 * math.pi * i / sides
            a1 = 2 * math.pi * (i + 1) / sides
            p0, p1 = (math.cos(a0), math.sin(a0)), (math.cos(a1), math.sin(a1))
            for (xy, z) in [(p0, z0), (p1, z0), (p1, z1), (p0, z0), (p1, z1), (p0, z1)]:
                # nudge z off the seam so the < 0.5 test is unambiguous
                zz = z if z != 0.5 else (0.4999 if half == 0 else 0.5001)
                verts.append((xy[0], xy[1], zz))
                norms.append((xy[0], xy[1], 0.0))
    return np.array(verts, np.float32), np.array(norms, np.float32)


def hull_triangles(points: np.ndarray):
    """Convex hull of the ligand positions, as a triangle soup with outward
    normals -- the coordination polyhedron."""
    from scipy.spatial import ConvexHull
    h = ConvexHull(points)
    c = points.mean(axis=0)
    tris, nrms = [], []
    for simplex in h.simplices:
        a, b, cc = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        n = np.cross(b - a, cc - a)
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        if np.dot(n, (a + b + cc) / 3.0 - c) < 0:
            a, b = b, a
            n = -n
        tris += [a, b, cc]
        nrms += [n, n, n]
    return np.array(tris, np.float32), np.array(nrms, np.float32)


def look_at(eye, target, up):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3], M[1, :3], M[2, :3] = s, u, -f
    M[:3, 3] = -M[:3, :3] @ np.asarray(eye, float)
    return M


def perspective(fovy_deg, aspect, znear, zfar):
    t = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    M = np.zeros((4, 4), np.float32)
    M[0, 0] = t / aspect
    M[1, 1] = t
    M[2, 2] = (zfar + znear) / (znear - zfar)
    M[2, 3] = 2 * zfar * znear / (znear - zfar)
    M[3, 2] = -1.0
    return M


# ---------------------------------------------------------------------------
# renderer
# ---------------------------------------------------------------------------

class Renderer:
    def __init__(self, gl: QOpenGLFunctions_3_3_Core):
        self.gl = gl
        self.progs = {}
        self._vaos = []
        self._bufs = []

    def program(self, name, vs, fs):
        p = QOpenGLShaderProgram()
        if not p.addShaderFromSourceCode(QOpenGLShader.Vertex, vs):
            raise RuntimeError(f"{name} vertex shader:\n{p.log()}")
        if not p.addShaderFromSourceCode(QOpenGLShader.Fragment, fs):
            raise RuntimeError(f"{name} fragment shader:\n{p.log()}")
        if not p.link():
            raise RuntimeError(f"{name} link:\n{p.log()}")
        self.progs[name] = p
        return p

    def buffer(self, data: np.ndarray):
        b = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        b.create(); b.bind()
        arr = np.ascontiguousarray(data, np.float32)
        b.allocate(arr.tobytes(), arr.nbytes)
        b.release()
        self._bufs.append(b)
        return b

    def vao(self):
        v = QOpenGLVertexArrayObject()
        v.create()
        self._vaos.append(v)
        return v

    def attrib(self, buf, loc, size, stride=0, offset=0, divisor=0):
        gl = self.gl
        buf.bind()
        gl.glEnableVertexAttribArray(loc)
        # Two PySide6 binding traps here. `normalized` is bound as an int, not a
        # bool. And the buffer offset must be a shiboken6.VoidPtr: an int and None
        # are rejected outright, but ctypes.c_void_p is ACCEPTED and silently
        # passes the wrong address, so the draw succeeds and renders nothing.
        gl.glVertexAttribPointer(loc, size, 0x1406, 0, stride,
                                 shiboken6.VoidPtr(offset))  # GL_FLOAT
        if divisor:
            gl.glVertexAttribDivisor(loc, divisor)
        buf.release()

    def fbo(self, size: QSize, n_extra=0, fmt=GL_RGBA16F):
        f = QOpenGLFramebufferObjectFormat()
        f.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
        o = QOpenGLFramebufferObject(size, f)
        for _ in range(n_extra):
            o.addColorAttachment(size, fmt)
        return o


def render(cif_path: str, out_png: str, width=1500, height=1100, ss=3,
           highlight_label: str | None = None, focus_label: str | None = None):
    t_start = time.perf_counter()
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QGuiApplication.instance() or QGuiApplication(sys.argv)
    surf = QOffscreenSurface()
    surf.setFormat(fmt)
    surf.create()
    ctx = QOpenGLContext()
    ctx.setFormat(fmt)
    if not ctx.create():
        raise RuntimeError("could not create an OpenGL 3.3 core context")
    ctx.makeCurrent(surf)

    gl = QOpenGLFunctions_3_3_Core()
    gl.initializeOpenGLFunctions()
    ver = ctx.format()
    print(f"  context      : OpenGL {ver.majorVersion()}.{ver.minorVersion()} core")

    t0 = time.perf_counter()
    st, cell, M, atoms, core = load_structure(cif_path)
    bonds = find_bonds(atoms, core, vmin=V_LIST if focus_label else V_BOND)
    t_struct = time.perf_counter() - t0
    print(f"  structure    : {st.name}  {st.spacegroup_hm}   "
          f"{len(core)} atoms in cell, {len(atoms)} in cluster, {len(bonds)} bonds "
          f"({t_struct*1000:.0f} ms)")

    if focus_label:
        # One site and everything it touches down to the tabulation threshold.
        # The point of the picture is that the outer contacts exist, and that
        # where the line falls between them is a choice, not a measurement.
        hits = [i for i, a in enumerate(atoms)
                if a[3] == focus_label and a[4] == (0, 0, 0)]
        if not hits:
            raise SystemExit("no site labelled " + repr(focus_label) + " in the cell")
        ci = hits[0]
        contacts = [(j if i == ci else i, v) for i, j, v in bonds if ci in (i, j)]
        contacts.sort(key=lambda t: -t[1])
        keep = [ci] + [j for j, v in contacts]
        remap = {o: n for n, o in enumerate(keep)}
        draw_atoms = [atoms[i] for i in keep]
        draw_bonds = [(0, remap[j], v) for j, v in contacts]
    else:
        # cell contents plus any periodic image that closes a bond
        keep = sorted({i for i, a in enumerate(atoms) if a[4] == (0, 0, 0)}
                      | {i for b in bonds for i in b[:2]})
        remap = {o: n for n, o in enumerate(keep)}
        draw_atoms = [atoms[i] for i in keep]
        draw_bonds = [(remap[i], remap[j], v) for i, j, v in bonds]

    pos = np.array([a[1] for a in draw_atoms], np.float32)
    els = [a[0] for a in draw_atoms]
    rad = np.array([RADII.get(e, 0.45) for e in els], np.float32)
    col = np.array([COLORS.get(e, (0.7, 0.7, 0.7)) for e in els], np.float32)

    centre = pos.mean(axis=0)
    # bounding sphere of what will actually be drawn, atom radii included
    radius = float((np.linalg.norm(pos - centre, axis=1) + rad).max())

    r = Renderer(gl)
    p_sphere = r.program("sphere", SPHERE_VS, SPHERE_FS)
    p_tube = r.program("tube", TUBE_VS, TUBE_FS)
    p_poly = r.program("poly", POLY_VS, POLY_FS)
    p_ssao = r.program("ssao", FS_QUAD_VS, SSAO_FS)
    p_comp = r.program("comp", FS_QUAD_VS, COMPOSITE_FS)
    print(f"  shaders      : {len(r.progs)} programs compiled and linked")

    quad = r.buffer(np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], np.float32))

    # --- sphere instances
    vao_s = r.vao(); vao_s.bind()
    r.attrib(quad, 0, 2)
    b_c = r.buffer(pos);  r.attrib(b_c, 1, 3, divisor=1)
    b_r = r.buffer(rad);  r.attrib(b_r, 2, 1, divisor=1)
    b_k = r.buffer(col);  r.attrib(b_k, 3, 3, divisor=1)
    vao_s.release()

    # --- bond instances
    tv, tn = split_tube(20)
    vao_t = r.vao(); vao_t.bind()
    b_tv = r.buffer(tv); r.attrib(b_tv, 0, 3)
    b_tn = r.buffer(tn); r.attrib(b_tn, 1, 3)
    if draw_bonds:
        A = np.array([pos[i] for i, j, v in draw_bonds], np.float32)
        B = np.array([pos[j] for i, j, v in draw_bonds], np.float32)
        # bond radius scales with bond valence -- the strength is visible
        # radius tracks bond valence, so a 0.03 v.u. contact reads as the
        # hairline it is next to a 0.5 v.u. bond
        BR = np.array([0.026 + 0.125 * min(v, 1.0) ** 0.70 for i, j, v in draw_bonds],
                      np.float32)
        def fade(c, v):
            # below the bond threshold a contact is still drawn, but greyed:
            # it is in the tabulation and out of the coordination number
            if v >= V_BOND:
                return c
            f = 0.30 + 0.70 * (v / V_BOND)
            return c * f + np.array([0.30, 0.32, 0.36], np.float32) * (1.0 - f)
        CA = np.array([fade(col[i], v) for i, j, v in draw_bonds], np.float32)
        CB = np.array([fade(col[j], v) for i, j, v in draw_bonds], np.float32)
        r.attrib(r.buffer(A), 2, 3, divisor=1)
        r.attrib(r.buffer(B), 3, 3, divisor=1)
        r.attrib(r.buffer(BR), 4, 1, divisor=1)
        r.attrib(r.buffer(CA), 5, 3, divisor=1)
        r.attrib(r.buffer(CB), 6, 3, divisor=1)
    vao_t.release()

    # --- coordination polyhedron around one highlighted site
    poly_n = 0
    vao_p = r.vao()
    hl = highlight_label or focus_label
    if hl:
        cand = [i for i, a in enumerate(draw_atoms)
                if a[3] == hl and a[4] == (0, 0, 0)]
        if cand:
            ci = cand[0]
            lig = np.array([pos[j] if i == ci else pos[i]
                            for i, j, v in draw_bonds
                            if ci in (i, j) and v >= V_BOND], np.float32)
            if len(lig) >= 4:
                tri, nrm = hull_triangles(lig)
                vao_p.bind()
                r.attrib(r.buffer(tri), 0, 3)
                r.attrib(r.buffer(nrm), 1, 3)
                vao_p.release()
                poly_n = len(tri)
                print(f"  polyhedron   : site {hl}, CN {len(lig)}, "
                      f"{poly_n // 3} hull faces")

    # --- camera: frame the bounding sphere exactly, with a small margin
    W, H = width * ss, height * ss
    FOV = 15.0
    aspect = W / H
    # the vertical half-angle is the binding one when the image is landscape
    half = math.radians(FOV) / 2.0
    dist = radius / math.sin(half) * 1.06
    look_dir = np.array([0.58, 0.40, 1.0])
    look_dir /= np.linalg.norm(look_dir)
    eye = centre + look_dir * dist
    view = look_at(eye, centre, (0, 1, 0))
    proj = perspective(FOV, aspect, max(0.05, dist - radius * 1.4), dist + radius * 1.4)
    light = np.array([0.42, 0.62, 0.66], np.float32)
    light /= np.linalg.norm(light)
    bg = np.array([0.086, 0.094, 0.110], np.float32)

    def setm(prog, name, m):
        # PySide6 wants a flat Sequence[float] here, not bytes. Our matrices are
        # numpy row-major acting on column vectors, so transpose = 1 converts to
        # the column-major order GLSL expects.
        loc = prog.uniformLocation(name)
        gl.glUniformMatrix4fv(loc, 1, 1, np.asarray(m, np.float32).ravel().tolist())

    # --- G-buffer pass
    gbuf = r.fbo(QSize(W, H), n_extra=2)
    gbuf.bind()
    gl.glDrawBuffers(3, [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2])
    gl.glViewport(0, 0, W, H)
    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    gl.glClear(0x4000 | 0x100)          # COLOR | DEPTH
    gl.glEnable(0x0B71)                 # DEPTH_TEST
    gl.glDisable(0x0BE2)                # BLEND

    t0 = time.perf_counter()
    p_sphere.bind()
    setm(p_sphere, "uView", view); setm(p_sphere, "uProj", proj)
    gl.glUniform3f(p_sphere.uniformLocation("uLight"), *(np.asarray(view[:3, :3]) @ light))
    vao_s.bind(); gl.glDrawArraysInstanced(0x0005, 0, 4, len(pos)); vao_s.release()  # TRIANGLE_STRIP
    p_sphere.release()

    if draw_bonds:
        p_tube.bind()
        setm(p_tube, "uView", view); setm(p_tube, "uProj", proj)
        gl.glUniform3f(p_tube.uniformLocation("uLight"), *(np.asarray(view[:3, :3]) @ light))
        vao_t.bind(); gl.glDrawArraysInstanced(0x0004, 0, len(tv), len(draw_bonds)); vao_t.release()
        p_tube.release()

    # --- transparent polyhedron, colour attachment only, depth test on / write off
    if poly_n:
        gl.glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])
        gl.glEnable(0x0BE2)
        gl.glBlendFunc(0x0302, 0x0303)   # SRC_ALPHA, ONE_MINUS_SRC_ALPHA
        gl.glDepthMask(False)
        gl.glDisable(0x0B44)             # CULL_FACE -- want both hull sides
        p_poly.bind()
        setm(p_poly, "uView", view); setm(p_poly, "uProj", proj)
        gl.glUniform3f(p_poly.uniformLocation("uLight"), *(np.asarray(view[:3, :3]) @ light))
        gl.glUniform3f(p_poly.uniformLocation("uColor"), 0.30, 0.78, 0.98)
        gl.glUniform1f(p_poly.uniformLocation("uAlpha"), 0.40)
        vao_p.bind(); gl.glDrawArrays(0x0004, 0, poly_n); vao_p.release()
        p_poly.release()
        gl.glDepthMask(True)
        gl.glDisable(0x0BE2)
    gl.glFinish()
    t_geom = time.perf_counter() - t0
    err = gl.glGetError()
    if err:
        print(f"  !! GL error after geometry pass: 0x{err:04X}")
    if DEBUG:
        gbuf.toImage().scaled(width, height, Qt.KeepAspectRatio,
                              Qt.SmoothTransformation).save("dbg_1_albedo.png")
    gbuf.release()

    tex = gbuf.textures()
    if DEBUG:
        print(f"  gbuf textures: {tex}")

    # --- SSAO pass
    aof = r.fbo(QSize(W, H))
    aof.bind()
    gl.glViewport(0, 0, W, H)
    gl.glDisable(0x0B71)
    gl.glClear(0x4000)
    t0 = time.perf_counter()
    p_ssao.bind()
    setm(p_ssao, "uProj", proj)
    gl.glUniform2f(p_ssao.uniformLocation("uRes"), W, H)
    gl.glUniform1f(p_ssao.uniformLocation("uRadius"), max(1.1, radius * 0.20))
    gl.glActiveTexture(0x84C0); gl.glBindTexture(0x0DE1, tex[1])
    gl.glUniform1i(p_ssao.uniformLocation("uPos"), 0)
    gl.glActiveTexture(0x84C1); gl.glBindTexture(0x0DE1, tex[2])
    gl.glUniform1i(p_ssao.uniformLocation("uNormal"), 1)
    vao_s.bind(); gl.glDrawArrays(0x0005, 0, 4); vao_s.release()
    p_ssao.release()
    gl.glFinish()
    t_ao = time.perf_counter() - t0
    err = gl.glGetError()
    if err:
        print(f"  !! GL error after SSAO pass: 0x{err:04X}")
    if DEBUG:
        aof.toImage().scaled(width, height, Qt.KeepAspectRatio,
                             Qt.SmoothTransformation).save("dbg_2_ao.png")
    aof.release()

    # --- composite
    outf = r.fbo(QSize(W, H))
    outf.bind()
    gl.glViewport(0, 0, W, H)
    gl.glClear(0x4000)
    p_comp.bind()
    gl.glUniform2f(p_comp.uniformLocation("uRes"), W, H)
    gl.glUniform1f(p_comp.uniformLocation("uFogNear"), dist - radius * 0.55)
    gl.glUniform1f(p_comp.uniformLocation("uFogFar"), dist + radius * 1.25)
    gl.glUniform3f(p_comp.uniformLocation("uBg"), *bg)
    gl.glActiveTexture(0x84C0); gl.glBindTexture(0x0DE1, tex[0])
    gl.glUniform1i(p_comp.uniformLocation("uColor"), 0)
    gl.glActiveTexture(0x84C1); gl.glBindTexture(0x0DE1, aof.texture())
    gl.glUniform1i(p_comp.uniformLocation("uAO"), 1)
    gl.glActiveTexture(0x84C2); gl.glBindTexture(0x0DE1, tex[1])
    gl.glUniform1i(p_comp.uniformLocation("uPos"), 2)
    vao_s.bind(); gl.glDrawArrays(0x0005, 0, 4); vao_s.release()
    p_comp.release()
    img = outf.toImage()
    outf.release()

    img = img.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    total = time.perf_counter() - t_start

    print(f"  geometry pass: {t_geom*1000:6.1f} ms  ({len(pos)} spheres, {len(draw_bonds)} bonds @ {W}x{H})")
    print(f"  framing      : bounding radius {radius:.2f} A, camera {dist:.2f} A, fov {FOV}deg")
    print(f"  SSAO pass    : {t_ao*1000:6.1f} ms  (24 samples)")
    print(f"  total        : {total*1000:6.0f} ms including CIF parse and shader compile")
    print(f"  wrote        : {out_png}  ({width}x{height})")
    ctx.doneCurrent()
    return out_png


if __name__ == "__main__":
    CIFDIR = str(Path(r"C:/Users/samso/Desktop/WSU_work/XRD/cif/Bi/cifs")) + "/"
    PO4 = CIFDIR + "7023719_BiPO4.cif"
    CIF = r"C:\Users\samso\Desktop\WSU_work\XRD\cif\Bi\cifs\1526458_Bi2O3.cif"
    print("alpha-Bi2O3 (Ivanov 2001, COD 1526458)")
    render(CIF, "proto_bi2o3.png", highlight_label="Bi1")
    print()
    print("alpha-Bi2O3 Bi1 site -- every contact down to 0.02 v.u.")
    render(CIF, "proto_bi1_site.png", focus_label="Bi1", width=1400, height=1400)
    print()
    print("BiPO4 -- two cations, two very different polyhedra")
    render(PO4, "proto_bipo4.png", highlight_label="Bi1", width=1500, height=1100)
