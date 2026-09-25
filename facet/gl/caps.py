"""What this machine's OpenGL can actually do, and which tier to run.

FACET must run on any Windows PC, including ones with no graphics hardware at
all. Three tiers cover that, chosen at startup from a real context rather than
from a guess:

``FULL``        OpenGL 3.3+ core. GLSL 330, screen-space ambient occlusion,
                supersampled export. What a machine with a GPU gets.
``COMPATIBLE``  OpenGL 3.0-3.2, which is what Qt's bundled software rasteriser
                offers. GLSL 130, attribute locations bound by name, ambient
                occlusion off by default.
``BASIC``       No usable OpenGL. Falls back to a QPainter renderer that sorts
                by depth and draws radial-gradient discs -- slower and flatter,
                but it works over remote desktop and on locked-down machines.

Measured facts that shaped this, from Qt 6.9 on Windows:

* Qt's ``opengl32sw.dll`` is **Mesa 11.2.2 llvmpipe**, which grants exactly
  OpenGL 3.0 / GLSL 1.30 whatever version is requested, and offers no
  instancing extension.
* Requesting a 3.3 core profile against the software rasteriser on a
  **QOffscreenSurface segfaults**. Software rendering must go through a real
  widget's context, so offscreen export on that tier renders into the widget's
  own framebuffer object instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

GL_VENDOR = 0x1F00
GL_RENDERER = 0x1F01
GL_VERSION = 0x1F02
GL_SHADING_LANGUAGE_VERSION = 0x8B8C
GL_MAX_TEXTURE_SIZE = 0x0D33
GL_MAX_SAMPLES = 0x8D57


class Tier(IntEnum):
    BASIC = 0
    COMPATIBLE = 1
    FULL = 2

    @property
    def label(self) -> str:
        return {Tier.BASIC: "Basic (no OpenGL)",
                Tier.COMPATIBLE: "Compatible (OpenGL 3.0)",
                Tier.FULL: "Full (OpenGL 3.3+)"}[self]


@dataclass
class Capabilities:
    tier: Tier
    major: int = 0
    minor: int = 0
    vendor: str = ""
    renderer: str = ""
    version: str = ""
    glsl: str = ""
    core_profile: bool = False
    software: bool = False
    max_texture: int = 0
    reason: str = ""

    @property
    def glsl_version(self) -> int:
        """The ``#version`` number the shaders must declare."""
        return 330 if self.tier is Tier.FULL else 130

    @property
    def supports_ssao(self) -> bool:
        # Possible on the compatible tier, but llvmpipe is fill-rate bound and
        # a full-screen 24-sample pass makes interaction unusable. Off unless
        # the user turns it on.
        return self.tier is Tier.FULL

    @property
    def supports_instancing(self) -> bool:
        return self.tier is Tier.FULL

    @property
    def supports_offscreen(self) -> bool:
        """Whether a standalone QOffscreenSurface context is safe here.

        It is not, on the software rasteriser: that combination crashes rather
        than failing, so export must use the widget's own context there.
        """
        return self.tier is Tier.FULL and not self.software

    def describe(self) -> str:
        if self.tier is Tier.BASIC:
            return f"{self.tier.label} -- {self.reason}"
        kind = "software" if self.software else "hardware"
        return (f"{self.tier.label} -- {self.renderer} ({kind}), "
                f"OpenGL {self.version}, GLSL {self.glsl}")


def detect(context) -> Capabilities:
    """Read capabilities from a current QOpenGLContext.

    The context must already be made current. Never raises: anything unexpected
    resolves to the BASIC tier, because a viewer that refuses to open is worse
    than one that draws flatly.
    """
    try:
        fmt = context.format()
        major, minor = fmt.majorVersion(), fmt.minorVersion()
    except Exception as exc:
        return Capabilities(Tier.BASIC, reason=f"no usable context ({exc})")

    functions, core = _load_functions(major, minor)
    if functions is None:
        return Capabilities(
            Tier.BASIC, major, minor,
            reason=f"OpenGL {major}.{minor} is below the 3.0 minimum")

    def get(name: int) -> str:
        try:
            return str(functions.glGetString(name) or "")
        except Exception:
            return ""

    renderer = get(GL_RENDERER)
    version = get(GL_VERSION)
    software = _looks_like_software(get(GL_VENDOR), renderer, version)

    tier = Tier.FULL if (major, minor) >= (3, 3) and core else Tier.COMPATIBLE
    caps = Capabilities(
        tier=tier, major=major, minor=minor,
        vendor=get(GL_VENDOR), renderer=renderer, version=version,
        glsl=get(GL_SHADING_LANGUAGE_VERSION),
        core_profile=core, software=software,
    )
    try:
        caps.max_texture = int(functions.glGetIntegerv(GL_MAX_TEXTURE_SIZE) or 0)
    except Exception:
        pass
    return caps


def _load_functions(major: int, minor: int):
    """The most capable function wrapper this context supports, and whether it
    is a core profile."""
    from PySide6.QtOpenGL import QOpenGLFunctions_3_0, QOpenGLFunctions_3_3_Core

    if (major, minor) >= (3, 3):
        fns = QOpenGLFunctions_3_3_Core()
        try:
            if fns.initializeOpenGLFunctions():
                return fns, True
        except Exception:
            pass
    if (major, minor) >= (3, 0):
        fns = QOpenGLFunctions_3_0()
        try:
            if fns.initializeOpenGLFunctions():
                return fns, False
        except Exception:
            pass
    return None, False


def _looks_like_software(vendor: str, renderer: str, version: str) -> bool:
    blob = f"{vendor} {renderer} {version}".lower()
    return any(k in blob for k in
               ("llvmpipe", "softpipe", "software", "swiftshader", "gdi generic",
                "microsoft basic render", "mesa offscreen"))


def request_format(prefer_core: bool = True):
    """The surface format to install before the application is constructed.

    Requesting 3.3 core is safe: a driver that cannot provide it hands back the
    highest it can, which the tier detection then reads. What is *not* safe is
    assuming the request was granted.
    """
    from PySide6.QtGui import QSurfaceFormat

    fmt = QSurfaceFormat()
    if prefer_core:
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
    # Multisampling is requested but routinely refused -- this driver grants 0
    # samples on the default framebuffer. Anti-aliasing therefore comes from
    # supersampling and from the analytic silhouettes of the sphere impostors,
    # neither of which depends on the driver agreeing.
    fmt.setSamples(4)
    return fmt
