"""GLSL, in two dialects.

The same programs are emitted for GLSL 330 core and GLSL 130, because the
compatible tier -- Qt's bundled Mesa llvmpipe -- offers only 1.30. The
differences are mechanical and are handled by :func:`emit`:

* 130 has no ``layout(location = N)``, so attribute locations are bound by name
  with ``glBindAttribLocation`` before linking. :data:`ATTRIBUTES` is the shared
  ordering both tiers use.
* 130 has no ``layout(location = N) out``; fragment outputs are bound with
  ``glBindFragDataLocation`` instead.
* ``texture()`` exists in both, but 130 needs ``#extension`` nothing -- it is
  fine as long as the sampler type is explicit.

Geometry is expanded on the CPU rather than instanced. Instancing is core only
from 3.3 and llvmpipe offers no extension for it, so an instanced path would
have to be written twice. Four vertices per atom is a rounding error on the
scale FACET draws, and one geometry path means one set of bugs.
"""
from __future__ import annotations

# Attribute names and the locations both tiers bind them to.
ATTRIBUTES = {
    "aCorner": 0,     # billboard corner, -1..1
    "aCenter": 1,     # sphere centre, world
    "aRadius": 2,
    "aColor": 3,
    "aId": 4,         # atom id, encoded for the picking pass
    "aPosition": 5,   # for meshes: tube, polyhedron, line
    "aNormal": 6,
    "aColorB": 7,     # tube: the second half's colour
    "aParam": 8,      # tube: fraction along the bond
}

FRAG_OUTPUTS = ("oColor", "oPos", "oNormal")


def emit(source: str, version: int) -> str:
    """Adapt one shader body to the requested GLSL version."""
    if version >= 330:
        header = "#version 330 core\n"
        body = source
    else:
        header = "#version 130\n"
        body = source
        # strip the layout qualifiers 130 cannot parse; locations are bound
        # by name instead, which produces the same result
        out_lines = []
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("layout(location"):
                line = line[line.index(")") + 1:].lstrip()
                line = "    " + line if line else line
            out_lines.append(line)
        body = "\n".join(out_lines)
    return header + body


# ---------------------------------------------------------------------------
# sphere impostors
# ---------------------------------------------------------------------------
# One camera-facing quad per atom; the sphere is intersected analytically in the
# fragment shader and gl_FragDepth is written. The silhouette is therefore exact
# at any magnification, there is no tessellation to choose, and the cost is set
# by fill rate rather than by triangle count.

SPHERE_VS = """
layout(location=0) in vec2 aCorner;
layout(location=1) in vec3 aCenter;
layout(location=2) in float aRadius;
layout(location=3) in vec3 aColor;
layout(location=4) in vec3 aId;

uniform mat4 uView;
uniform mat4 uProj;

out vec3 vCenterView;
out float vRadius;
out vec3 vColor;
out vec3 vRayView;
out vec3 vId;

void main() {
    vec4 centre = uView * vec4(aCenter, 1.0);
    vCenterView = centre.xyz;
    vRadius = aRadius;
    vColor = aColor;
    vId = aId;
    // 1.15 oversize: with a perspective projection the silhouette of a sphere
    // is slightly larger than its radius, and a tight quad clips the edge.
    vec3 p = centre.xyz + vec3(aCorner * aRadius * 1.15, 0.0);
    vRayView = p;
    gl_Position = uProj * vec4(p, 1.0);
}
"""

SPHERE_FS = """
in vec3 vCenterView;
in float vRadius;
in vec3 vColor;
in vec3 vRayView;
in vec3 vId;

uniform mat4 uProj;
uniform vec3 uLight;
uniform int uPicking;

layout(location=0) out vec4 oColor;
layout(location=1) out vec4 oPos;
layout(location=2) out vec4 oNormal;

void main() {
    vec3 rd = normalize(vRayView);
    vec3 oc = -vCenterView;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - vRadius * vRadius;
    float h = b * b - c;
    if (h < 0.0) discard;

    float t = -b - sqrt(h);
    vec3 p = rd * t;
    vec3 n = normalize(p - vCenterView);

    vec4 clip = uProj * vec4(p, 1.0);
    gl_FragDepth = 0.5 * (clip.z / clip.w) + 0.5;

    if (uPicking == 1) {
        oColor = vec4(vId, 1.0);
        oPos = vec4(p, 1.0);
        oNormal = vec4(n, 1.0);
        return;
    }

    float diff = max(dot(n, uLight), 0.0);
    vec3 half_v = normalize(uLight + vec3(0.0, 0.0, 1.0));
    float spec = pow(max(dot(n, half_v), 0.0), 48.0);
    float rim = pow(1.0 - max(dot(n, vec3(0.0, 0.0, 1.0)), 0.0), 2.4);
    vec3 col = vColor * (0.26 + 0.74 * diff)
             + vec3(0.30) * spec
             + vColor * 0.16 * rim;

    oColor = vec4(col, 1.0);
    oPos = vec4(p, 1.0);
    oNormal = vec4(n, 1.0);
}
"""

# ---------------------------------------------------------------------------
# bonds: low-poly tubes, split at the midpoint so each half takes its own colour
# ---------------------------------------------------------------------------

TUBE_VS = """
layout(location=5) in vec3 aPosition;
layout(location=6) in vec3 aNormal;
layout(location=3) in vec3 aColor;
layout(location=7) in vec3 aColorB;
layout(location=8) in float aParam;

uniform mat4 uView;
uniform mat4 uProj;

out vec3 vPosView;
out vec3 vNormView;
out vec3 vColor;

void main() {
    vec4 pv = uView * vec4(aPosition, 1.0);
    vPosView = pv.xyz;
    vNormView = normalize(mat3(uView) * aNormal);
    vColor = aParam < 0.5 ? aColor : aColorB;
    gl_Position = uProj * pv;
}
"""

TUBE_FS = """
in vec3 vPosView;
in vec3 vNormView;
in vec3 vColor;

uniform vec3 uLight;
uniform int uPicking;
uniform vec3 uPickId;

layout(location=0) out vec4 oColor;
layout(location=1) out vec4 oPos;
layout(location=2) out vec4 oNormal;

void main() {
    vec3 n = normalize(vNormView);
    if (uPicking == 1) {
        oColor = vec4(uPickId, 1.0);
        oPos = vec4(vPosView, 1.0);
        oNormal = vec4(n, 1.0);
        return;
    }
    float diff = max(dot(n, uLight), 0.0);
    vec3 half_v = normalize(uLight + vec3(0.0, 0.0, 1.0));
    float spec = pow(max(dot(n, half_v), 0.0), 40.0);
    vec3 col = vColor * (0.30 + 0.70 * diff) + vec3(0.22) * spec;
    oColor = vec4(col, 1.0);
    oPos = vec4(vPosView, 1.0);
    oNormal = vec4(n, 1.0);
}
"""

# ---------------------------------------------------------------------------
# coordination polyhedra, forward-rendered with blending after the opaque pass
# ---------------------------------------------------------------------------

POLY_VS = """
layout(location=5) in vec3 aPosition;
layout(location=6) in vec3 aNormal;

uniform mat4 uView;
uniform mat4 uProj;

out vec3 vNormView;
out vec3 vPosView;

void main() {
    vec4 pv = uView * vec4(aPosition, 1.0);
    vPosView = pv.xyz;
    vNormView = normalize(mat3(uView) * aNormal);
    gl_Position = uProj * pv;
}
"""

POLY_FS = """
in vec3 vNormView;
in vec3 vPosView;

uniform vec3 uColor;
uniform float uAlpha;
uniform vec3 uLight;

layout(location=0) out vec4 oColor;

void main() {
    vec3 n = normalize(vNormView);
    if (!gl_FrontFacing) n = -n;
    float diff = max(dot(n, uLight), 0.0);
    // a fresnel term so the rim still reads where the face is nearly edge-on,
    // which is what makes a transparent polyhedron look like a solid
    float fres = pow(1.0 - abs(dot(n, vec3(0.0, 0.0, 1.0))), 2.0);
    vec3 col = uColor * (0.45 + 0.55 * diff) + vec3(0.25) * fres;
    oColor = vec4(col, clamp(uAlpha + 0.42 * fres, 0.0, 0.94));
}
"""

# ---------------------------------------------------------------------------
# unit-cell edges and measurement lines
# ---------------------------------------------------------------------------

LINE_VS = """
layout(location=5) in vec3 aPosition;
uniform mat4 uView;
uniform mat4 uProj;
out vec3 vPosView;
void main() {
    vec4 pv = uView * vec4(aPosition, 1.0);
    vPosView = pv.xyz;
    gl_Position = uProj * pv;
}
"""

# Lines write the whole G-buffer, not just colour. The composite treats an
# unwritten position attachment as background, so a shader that emitted only a
# colour would have its pixels discarded -- which is exactly how the unit cell
# came to be invisible while being drawn correctly.
LINE_FS = """
in vec3 vPosView;
uniform vec3 uColor;
uniform float uAlpha;
layout(location=0) out vec4 oColor;
layout(location=1) out vec4 oPos;
layout(location=2) out vec4 oNormal;
void main() {
    oColor = vec4(uColor, uAlpha);
    oPos = vec4(vPosView, 1.0);
    oNormal = vec4(0.0, 0.0, 1.0, 1.0);
}
"""

# ---------------------------------------------------------------------------
# full-screen passes
# ---------------------------------------------------------------------------

QUAD_VS = """
layout(location=0) in vec2 aCorner;
out vec2 vUv;
void main() {
    vUv = aCorner * 0.5 + 0.5;
    gl_Position = vec4(aCorner, 0.0, 1.0);
}
"""

SSAO_FS = """
in vec2 vUv;

uniform sampler2D uPos;
uniform sampler2D uNormal;
uniform mat4 uProj;
uniform vec2 uRes;
uniform float uRadius;

layout(location=0) out vec4 oAO;

const int NS = 24;

vec3 kern(int i) {
    float a = float(i) * 2.39996323;         // golden angle
    float r = sqrt((float(i) + 0.5) / float(NS));
    // leaning towards the tangent plane rather than the normal: on a scene of
    // convex spheres the occlusion signal lives in the grazing directions,
    // and a normal-biased kernel measures almost nothing
    float z = 0.10 + 0.50 * r;
    return normalize(vec3(cos(a) * r, sin(a) * r, z)) * (0.25 + 0.75 * r);
}

void main() {
    vec4 P = texture(uPos, vUv);
    if (P.w < 0.5) { oAO = vec4(1.0); return; }

    vec3 p = P.xyz;
    vec3 n = normalize(texture(uNormal, vUv).xyz);
    vec3 t = abs(n.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 bx = normalize(cross(t, n));
    mat3 TBN = mat3(bx, cross(n, bx), n);

    float occ = 0.0;
    for (int i = 0; i < NS; ++i) {
        vec3 sp = p + TBN * kern(i) * uRadius;
        vec4 cl = uProj * vec4(sp, 1.0);
        vec2 uv = (cl.xy / cl.w) * 0.5 + 0.5;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) continue;
        vec4 SP = texture(uPos, uv);
        if (SP.w < 0.5) continue;
        float rangeCheck = smoothstep(0.0, 1.0, uRadius / abs(p.z - SP.z));
        if (SP.z >= sp.z + 0.015) occ += rangeCheck;
    }
    oAO = vec4(vec3(clamp(1.0 - occ / float(NS), 0.0, 1.0)), 1.0);
}
"""

COMPOSITE_FS = """
in vec2 vUv;

uniform sampler2D uColor;
uniform sampler2D uAO;
uniform sampler2D uPos;
uniform vec2 uRes;
uniform float uFogNear;
uniform float uFogFar;
uniform float uFogAmount;
uniform vec3 uBg;
uniform int uUseAO;
uniform int uOutline;

layout(location=0) out vec4 oCol;

void main() {
    vec4 P = texture(uPos, vUv);
    vec3 c = texture(uColor, vUv).rgb;

    if (P.w < 0.5) {
        // Nothing opaque here, but transparent geometry -- a coordination
        // polyhedron overhanging the edge of the structure -- may have blended
        // into the colour attachment, which was cleared to zero. Un-premultiply
        // it and lay it over the background rather than discarding it.
        vec4 t = texture(uColor, vUv);
        oCol = vec4(mix(uBg, t.rgb / max(t.a, 1e-4), clamp(t.a, 0.0, 1.0)), 1.0);
        return;
    }

    if (uUseAO == 1) {
        float ao = 0.0;
        for (int dy = -2; dy <= 2; ++dy)
            for (int dx = -2; dx <= 2; ++dx)
                ao += texture(uAO, vUv + vec2(dx, dy) / uRes).r;
        ao /= 25.0;
        // ao returns close to 1 over most of a convex scene; a power curve
        // expands the crevices into a range the eye can see
        c *= mix(0.18, 1.0, pow(clamp(ao, 0.0, 1.0), 3.2));
    }

    if (uOutline == 1) {
        float edge = 0.0;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                vec4 Q = texture(uPos, vUv + vec2(dx, dy) * 1.6 / uRes);
                edge = max(edge, abs(Q.z - P.z) * (Q.w > 0.5 ? 1.0 : 0.0));
            }
        c *= mix(0.45, 1.0, 1.0 - smoothstep(0.03, 0.16, edge));
    }

    float f = clamp((-P.z - uFogNear) / max(uFogFar - uFogNear, 1e-4), 0.0, 1.0);
    c = mix(c, uBg, f * uFogAmount);

    oCol = vec4(pow(c, vec3(1.0 / 1.9)), 1.0);
}
"""

# A composite for the compatible tier: no ambient-occlusion texture lookup and
# no nine-tap outline, because llvmpipe is fill-rate bound and those two passes
# alone cost more than everything else drawn.
COMPOSITE_SIMPLE_FS = """
in vec2 vUv;

uniform sampler2D uColor;
uniform sampler2D uPos;
uniform float uFogNear;
uniform float uFogFar;
uniform float uFogAmount;
uniform vec3 uBg;

layout(location=0) out vec4 oCol;

void main() {
    vec4 P = texture(uPos, vUv);
    if (P.w < 0.5) {
        vec4 t = texture(uColor, vUv);
        oCol = vec4(mix(uBg, t.rgb / max(t.a, 1e-4), clamp(t.a, 0.0, 1.0)), 1.0);
        return;
    }
    vec3 c = texture(uColor, vUv).rgb;
    float f = clamp((-P.z - uFogNear) / max(uFogFar - uFogNear, 1e-4), 0.0, 1.0);
    c = mix(c, uBg, f * uFogAmount);
    oCol = vec4(pow(c, vec3(1.0 / 1.9)), 1.0);
}
"""
