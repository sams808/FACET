# Bundled example structures

Six crystal structures ship with FACET so that the worked examples in the
manual can be followed on a fresh installation, with nothing to download.

All six were obtained from the **Crystallography Open Database**
(https://www.crystallography.net/cod/), which places its contents in the public
domain under the CC0 1.0 Universal dedication. They are redistributed here
unchanged. No restriction is placed on passing FACET on with them included.

The underlying determinations are the work of their authors, cited below. COD
entries are credited to the original publication, and citing the paper rather
than the database is the courtesy the COD asks for.

| File | Phase | COD | Determination |
|---|---|---|---|
| `quartz_SiO2_cod9013321.cif` | Quartz, SiO₂ | 9013321 | Antao, S. M., Hassan, I., Wang, J. *et al.* (2008). *The Canadian Mineralogist* **46**, 1501. |
| `eulytite_Bi4SiO4_3_cod9012894.cif` | Eulytite, Bi₄(SiO₄)₃ | 9012894 | Barbier, J., Greedan, J. E., Asaro, T. *et al.* (1990). *European Journal of Solid State and Inorganic Chemistry* **27**, 855. |
| `senarmontite_Sb2O3_cod9009747.cif` | Senarmontite, Sb₂O₃ | 9009747 | Whitten, A. E., Dittrich, B., Spackman, M. A. *et al.* (2004). *Dalton Transactions*, 23. |
| `valentinite_Sb2O3_cod9007587.cif` | Valentinite, Sb₂O₃ | 9007587 | Svensson, C. (1974). *Acta Crystallographica Section B* **30**, 458. |
| `cryolite_Na3AlF6_cod9004097.cif` | Cryolite, Na₃AlF₆ | 9004097 | Hawthorne, F. C. & Ferguson, R. B. (1975). *The Canadian Mineralogist* **13**, 377. |
| `bismuth_phosphate_BiPO4_cod9008088.cif` | Ximengite, BiPO₄ | 9008088 | Mooney-Slater, R. C. L. (1962). *Zeitschrift für Kristallographie* **117**, 371. |

## Why these six

They are chosen to disagree with one another, so that the manual can show what
a settled coordination number looks like and what an unsettled one looks like
without leaving the bundled set.

- **Quartz** — a coordination number nobody argues about; the baseline.
- **Eulytite** — two sites in one structure, one as clean as quartz and one
  with no gap for a cutoff to fall in.
- **Senarmontite** and **valentinite** — the same compound, Sb₂O₃, twice; the
  two do not report the same coordination number, and neither is wrong.
- **Cryolite** — a fluoride, where a cutoff chosen for oxides has no standing.
  Its Na2 site has a plateau about a tenth of a decade wide: a coordination
  number that belongs to the threshold rather than to the structure.
- **BiPO₄** — the Bi(III) of eulytite in a different host, for separating what
  belongs to the ion from what belongs to the compound.
