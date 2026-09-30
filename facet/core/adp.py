"""Anisotropic displacement parameters, as a tensor and as a shape.

A CIF gives them as six numbers per site, ``U_11`` to ``U_23``, and they are not
Cartesian: they are components on the reciprocal basis, defined by the
temperature factor

    T = exp(-2 pi^2 sum_ij U_ij h_i h_j a*_i a*_j)

which means nothing about their size or sign can be read off directly. ``U_11``
larger than ``U_22`` does not say the atom moves further along *a* than along
*b*, and in a monoclinic or triclinic cell the off-diagonal terms are not even
in the units one would guess. Everything physical -- the principal directions,
how far the atom moves along each of them, whether the tensor describes an
ellipsoid at all -- lives in the Cartesian tensor, so that is what this module
computes.

Why it is here rather than only in the renderer: the displacement parameters are
a statement about how well the structure was determined. A tensor with a
non-positive eigenvalue is not a flat or thin ellipsoid, it is not an ellipsoid,
and a site carrying one was refined into a region the data did not support. That
bears directly on whether a coordination number taken from the same file means
anything -- which is a measurement to report, not a verdict to pass, so
:mod:`facet.core.quality` states it and leaves the judgement to the reader.

The conventions this module has to keep straight:

* ``U`` is a mean-square displacement in square angstroms; ``B`` is
  ``8 pi^2 U`` and is what older files give. gemmi converts ``B_ij`` loops to
  ``U_ij`` on the way in, so everything below is ``U``.
* The six numbers are ordered ``U11 U22 U33 U12 U13 U23``, which is the CIF
  order and gemmi's, and is *not* Voigt order -- Voigt would put ``U23``
  before ``U13``. Getting that wrong swaps two off-diagonal terms and tilts
  every ellipsoid in the wrong direction, while leaving ``U_eq`` correct,
  which is exactly the kind of error that survives a plausibility check.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# The scale factor from a root-mean-square displacement to the surface enclosing
# a given fraction of the distribution, for three degrees of freedom: the square
# root of the chi-squared quantile. 50% is the crystallographic default and the
# one every other program draws by default; the others are what a paper asks for
# when the 50% ellipsoids come out too small to see.
PROBABILITIES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
DEFAULT_PROBABILITY = 0.50

# Below this, an ellipsoid drawn at any probability is a dot, and the site was
# almost certainly refined with a fixed placeholder rather than measured.
NEGLIGIBLE_U = 1e-6


def scale_for(probability: float = DEFAULT_PROBABILITY) -> float:
    """Multiplier on the principal r.m.s. displacements for a probability.

    ``sqrt(chi2.ppf(p, 3))``: the radius, in units of the r.m.s. displacement
    along each axis, of the surface enclosing that fraction of a trivariate
    normal distribution. 50% gives 1.5382, which is why a "50% ellipsoid" is
    half again as large as the r.m.s. surface rather than smaller than it.
    """
    p = float(probability)
    if not 0.0 < p < 1.0:
        raise ValueError(f"a probability must be between 0 and 1, not {p}")
    from scipy.stats import chi2

    return float(math.sqrt(chi2.ppf(p, 3)))


def reciprocal_lengths(orth: np.ndarray) -> np.ndarray:
    """``(a*, b*, c*)`` from the orthogonalisation matrix.

    Taken from ``orth`` rather than from the cell lengths and angles so that
    there is one definition of the cell in play. The reciprocal basis vectors
    are the rows of the inverse of the matrix whose columns are the direct
    ones, which is what ``orth`` is.
    """
    inverse = np.linalg.inv(np.asarray(orth, float))
    return np.linalg.norm(inverse, axis=1)


def cartesian_tensor(orth: np.ndarray, u_cif) -> np.ndarray:
    """The Cartesian displacement tensor, in square angstroms.

    ``U_cart = (A N) U (A N)^T`` where ``A`` is the orthogonalisation matrix and
    ``N`` is the diagonal matrix of reciprocal cell lengths. ``N`` is what turns
    the dimensionless CIF components into a tensor on the direct Cartesian axes;
    leaving it out is the classic error, and in a cell whose axes are similar in
    length it gives an answer that is wrong by a constant factor and therefore
    looks plausible.
    """
    u = np.asarray(u_cif, float).reshape(6)
    tensor = np.array([[u[0], u[3], u[4]],
                       [u[3], u[1], u[5]],
                       [u[4], u[5], u[2]]], float)
    m = np.asarray(orth, float) @ np.diag(reciprocal_lengths(orth))
    return m @ tensor @ m.T


@dataclass(frozen=True)
class Ellipsoid:
    """What a site's displacement tensor is, in Cartesian terms.

    ``eigenvalues`` are mean-square displacements along ``axes``, in ascending
    order; the columns of ``axes`` are the corresponding directions. Nothing
    here is a drawing instruction: the renderer scales by
    :func:`scale_for` at whatever probability is being displayed.
    """

    u_cart: np.ndarray                   # 3x3, symmetric, square angstroms
    eigenvalues: np.ndarray              # 3, ascending
    axes: np.ndarray                     # 3x3, columns are the directions

    @property
    def u_equivalent(self) -> float:
        """``U_eq``: one third of the trace, the isotropic equivalent.

        The quantity a file reports as ``U_eq`` and the one comparable with
        ``U_iso``. It is a trace, so it survives a wrong choice of basis that
        the individual components do not -- which is why agreement on ``U_eq``
        alone does not confirm the conversion.
        """
        return float(np.trace(self.u_cart) / 3.0)

    @property
    def rms(self) -> np.ndarray:
        """Root-mean-square displacement along each principal axis, in Å.

        Negative eigenvalues have no root; they come back as ``nan`` rather
        than as a number that would let a caller carry on as if the tensor
        described an ellipsoid. :attr:`is_ellipsoid` is the question to ask.
        """
        values = np.asarray(self.eigenvalues, float)
        return np.where(values > 0, np.sqrt(np.abs(values)), np.nan)

    @property
    def is_ellipsoid(self) -> bool:
        """Whether the tensor describes a surface at all.

        A displacement tensor is a covariance matrix and must be positive
        definite. One that is not is usually called a "non-positive-definite"
        or NPD atom; it describes a hyperboloid, which is not a shape an atom
        can have, and it means the refinement placed the site where the data
        did not constrain it.
        """
        return bool(np.all(np.asarray(self.eigenvalues, float) > 0))

    @property
    def anisotropy(self) -> float:
        """Longest principal axis over shortest, as a ratio of r.m.s. lengths.

        1 is a sphere. ``inf`` if the tensor is not an ellipsoid, because the
        ratio is then meaningless rather than large.
        """
        if not self.is_ellipsoid:
            return float("inf")
        values = np.asarray(self.eigenvalues, float)
        return float(math.sqrt(values[2] / values[0]))

    def radii(self, probability: float = DEFAULT_PROBABILITY) -> np.ndarray:
        """The three semi-axis lengths of the drawn ellipsoid, in Å."""
        return self.rms * scale_for(probability)

    def transform(self, probability: float = DEFAULT_PROBABILITY) -> np.ndarray:
        """The matrix carrying a unit sphere onto the drawn ellipsoid.

        ``axes @ diag(radii)``. Right-multiplying a unit sphere's vertices by
        its transpose, or left-multiplying its column vectors, gives the
        surface; the same matrix's inverse transpose carries the normals.
        """
        return np.asarray(self.axes, float) @ np.diag(self.radii(probability))


def ellipsoid(orth: np.ndarray, u_cif) -> Ellipsoid:
    """Diagonalise a site's CIF displacement parameters in Cartesian axes."""
    u_cart = cartesian_tensor(orth, u_cif)
    # eigh, not eig: the tensor is symmetric by construction, and eigh returns
    # real values in ascending order with orthonormal vectors. eig would give
    # complex values with a vanishing imaginary part and no guaranteed order.
    values, vectors = np.linalg.eigh(u_cart)
    return Ellipsoid(u_cart=u_cart, eigenvalues=values, axes=vectors)


def for_site(cell, site) -> Ellipsoid | None:
    """The ellipsoid for a site, or ``None`` if the file gave no tensor.

    A site with only ``U_iso`` is deliberately not turned into a sphere here.
    That would be a drawing decision dressed as a measurement, and the two
    cases have to stay distinguishable: a file that refined every site
    anisotropically says something different from one that did not.
    """
    u = getattr(site, "u_aniso", None)
    if u is None:
        return None
    u = np.asarray(u, float).reshape(6)
    if not np.all(np.isfinite(u)) or not np.any(u):
        return None
    return ellipsoid(cell.orth, u)


def sphere(u_iso: float) -> Ellipsoid:
    """The isotropic case as the same type, for a caller that wants one shape.

    Used by the renderer, which draws every atom the same way and needs a
    tensor for each; not used by anything that reports what the file said.
    """
    value = max(float(u_iso), 0.0)
    return Ellipsoid(u_cart=np.eye(3) * value,
                     eigenvalues=np.full(3, value),
                     axes=np.eye(3))
