# Optical basicity from Python and the command line

`facet.basicity` gives the optical-basicity calculations of FACET without the
window: from a script, a notebook or a terminal. It is one part of the FACET
library; the rest (coordination analysis of crystals, the analyses of MD
models, figures and pictures) is described in
[library_manual.md](library_manual.md). It is part of the `facet`
package, imports no window toolkit, and returns the same numbers as the
Basicity tab, the Site rows, the MD analysis and the basicity maps.

```
pip install git+https://github.com/sams808/FACET          # the package
pip install matplotlib                                    # for the figures
```

Every command below is `python -m facet.basicity ...`; after a `pip install`
the same commands are also available as `facet-basicity ...`.

## What is computed

| Quantity | Definition | Source |
|---|---|---|
| γ of a cation | 1.36 (χ − 0.26), or a tabulated value | Duffy and Ingram 1976; four scales, below |
| Λ of an oxide | 1 / γ | Duffy and Ingram 1976 |
| Λ of a fluoride | 1 / (2.3 γ) | Duffy 1989 |
| Λ of a composition | Σ X_c Λ_c, X_c the share of the anion charge component c carries | Duffy and Ingram 1976, eq. 4; Duffy 1989, eq. 4 |
| λ of one anion | [1 − Σ (s_j / \|z\|)(1 − 1/γ_j)] / f over its bonded contacts, s_j the bond valence, f = 1 for oxide and 2.3 for fluoride | Duffy and Ingram 1976, section 1.4, with bond valences as the shares |
| λ about a cation | bond-valence-weighted mean λ of the anions bonded to it | — |
| α(O²⁻) | 1.67 / (1.67 − Λ), in Å³ | Duffy 1989 (Phys. Chem. Glasses 30, 1) |
| probe-ion band | ν = ν_free − shift × Λ for Tl⁺ (55 300, 18 300), Pb²⁺ (60 700, 31 000), Bi³⁺ (56 000, 28 800) cm⁻¹ | Duffy and Ingram 1976, eqs. 1–3 |
| field λ(r) | mean λ of the anions a probe cation at r would bond, weighted by the bond valence it would form with each | — |
| C(r) | ⟨δλ_i δλ_j⟩(r) / ⟨δλ²⟩ over anion pairs a distance r apart | — |

The scales of γ are `pauling` (Pauling's 1960 electronegativity through the
relation; the default), `duffy` (Duffy 1996, Table 1), `lebouteiller`
(Lebouteiller and Courtine 1998, Table 2) and `dimitrov` (Dimitrov and Sakka
1996, Table I). A cation a table does not list takes the relation, and every
value says which it is. No result is graded: each is a number with the scale,
the threshold and the notes that produced it.

## In Python

```python
from facet import basicity as fb

# a composition, as written on a batch sheet
c = fb.composition("30Na2O-70SiO2")              # moles; unit="wt" for weights
c.value                                          # 0.596
c.polarisability, c.band("Pb2+")                 # 1.555 Å3, 42225 cm-1
print(c)                                         # the table, component by component
x, lam = fb.series("Na2O", "SiO2", steps=21)     # Λ along a join

# a crystal structure (CIF, VESTA, POSCAR, SHELX, PDB, CrystalMaker)
s = fb.structure("albite.cif", scale="duffy", formers=["Si", "Al"])
s.bulk, s.value, s.difference                    # Λ composition, mean λ, their difference
s.anions()                                       # one row per anion site
s.site("O1").shares                              # each cation's share and lowering
s.linkages()                                     # λ by what the anion is bonded to
s.to_csv("sites.csv"); s.to_dict()

# one frame of an MD model (LAMMPS, extended XYZ, DL_POLY, VASP, ...)
g = fb.frame("glass.data", formers=["Si", "Al"],
             type_map={1: "O", 2: "Ca", 3: "Na", 4: "Al", 5: "Si"})
g.values                                         # λ per atom, NaN for a cation
g.environment                                    # per cation, the mean λ of its anions
g.by_class()                                     # free, NBO, BO, tricluster
g.linkages()                                     # Si-O-Si, Si-O-Na3, ...
g.correlation(r_max=10)                          # C(r)
g.band("Bi")                                     # predicted band of each Bi3+
for f in fb.frames("run.lammpstrj", step=10):    # a trajectory, frame by frame
    print(f.index, f.value)

# the field a probe cation senses, for a crystal or a frame
field = g.field("Ca", grid=1.0, smooth=2.0)
field.values                                     # the grid, NaN where undefined
field.slice("c", 0.5)                            # a plane, with the atoms near it
field.to_cube("lambda_Ca.cube")                  # opens in VESTA, VMD, OVITO

# single values and the published relations
fb.oxide("Si", 4), fb.oxide("Ca", 2, anion="F")
fb.moderator("P", 5, "duffy")
fb.probe_band(0.60, "Bi3+"), fb.basicity_from_band(44600, "Pb2+")
fb.oxide_polarisability(0.60)
```

`fb.load(path)` picks `structure` or `frame` by what the file is.

### Figures

`fb.plot` needs matplotlib. Each function returns the figure; `save="name.png"`
(or `.pdf`, `.svg`) writes it and `show=True` opens a window.

| Call | Draws |
|---|---|
| `fb.plot.series("Na2O", "SiO2")` | Λ along a join, one line per scale |
| `fb.plot.sites(s)` | λ of every anion site of a crystal, with Λ and the mean |
| `fb.plot.histogram(g, by="class")` | the distribution of λ, by class, element or linkage |
| `fb.plot.linkages(g)` | mean and spread of λ per linkage |
| `fb.plot.slice(field, "c", 0.5)` | a section of the field with the atoms near it |
| `fb.plot.correlation(g)` | C(r) |
| `fb.plot.band(g, "Bi")` | the predicted probe-ion band of each atom of an element |

## From a terminal

```
python -m facet.basicity scales
python -m facet.basicity oxide Si
python -m facet.basicity composition "30Na2O-70SiO2" --scale all
python -m facet.basicity composition "15Na2O 85SiO2" --wt --json
python -m facet.basicity series Na2O SiO2 --steps 11 --csv join.csv
python -m facet.basicity band 0.60
python -m facet.basicity band 44600 --from-band --ion Pb2+

python -m facet.basicity structure albite.cif --formers Si,Al --csv sites.csv
python -m facet.basicity md glass.data --formers Si,Al --correlation
python -m facet.basicity field glass.data --probe Ca --smooth 2 --cube lambda.cube
```

Quick figures: with `-o FILE` the figure is written, without it a window opens.

```
python -m facet.basicity plot series Na2O SiO2 -o join.png
python -m facet.basicity plot sites albite.cif --formers Si,Al
python -m facet.basicity plot histogram glass.data --formers Si,Al --by class
python -m facet.basicity plot linkages glass.data --formers Si,Al -o linkages.pdf
python -m facet.basicity plot slice glass.data --probe Ca --smooth 2 --at 0.5
python -m facet.basicity plot correlation glass.data
python -m facet.basicity plot band glass.data --probe Bi
python -m facet.basicity view glass.data
```

`view` opens the file in the FACET window with the atoms coloured by
microscopic basicity.

Options shared by the commands that read a file:

| Option | Meaning |
|---|---|
| `--scale` | `pauling`, `duffy`, `lebouteiller`, `dimitrov` |
| `--v-bond` | the bond threshold in valence units (default 0.075) |
| `--formers Si,Al` | the network formers: the free / NBO / BO split and the linkage names |
| `--frame N` | MD: the frame, from 0, negative from the end (default the last) |
| `--ox Fe=2` | MD: oxidation states that differ from the usual ones |
| `--type-map 1=Si,2=O` | MD: LAMMPS types or labels to elements |
| `--as crystal` / `--as md` | how to read a file whose kind is not clear from it |
| `--json`, `--csv FILE` | a mapping instead of tables; the rows as a CSV file |

A command exits with 0 on success and 2 when something could not be read or
computed, with the reason on the error stream.

## Writing a composition

Amounts go in front of their formulas, or after them, with any of `-`, `,`,
`;`, `:`, `+`, `·` or spaces between: `30Na2O-70SiO2`, `Na2O: 30, SiO2: 70`,
`0.3Na2O·0.7SiO2`. A formula with no amount in front counts one
(`Na2O-2SiO2`). Parentheses work (`Ca3(PO4)2`). A component's cation takes the
oxidation state its formula states (FeO is Fe²⁺, Fe2O3 is Fe³⁺); a mixed-valence
oxide is written as its parts. Fluoride components are counted with their own
cation, as in Duffy's 1989 equation. A composition that cannot be read is
refused with the reason.

## Reading the linkage names

The anion sits in the middle, the named formers on its left, the other bonded
cations on its right, each with its count when above one: `Si-O-Si`,
`Al-O-Si`, `Si-O-Na3`, `Si2-O-Ca` (two Si and one Ca), `O-Ca4 Na2` (no former).
A contact counts when its bond valence is above the threshold.

## References

- J. A. Duffy and M. D. Ingram, J. Non-Cryst. Solids 21 (1976) 373.
- J. A. Duffy, J. Non-Cryst. Solids 109 (1989) 35.
- J. A. Duffy, Phys. Chem. Glasses 30 (1989) 1.
- J. A. Duffy, J. Chem. Educ. 73 (1996) 1138.
- A. Lebouteiller and P. Courtine, J. Solid State Chem. 137 (1998) 94.
- V. Dimitrov and S. Sakka, J. Appl. Phys. 79 (1996) 1736.
