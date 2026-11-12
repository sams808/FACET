# Reference structures for the tests

Fifteen crystal structures that the tests name one by one. They are kept in
the repository so that a checkout on any machine, and the continuous
integration, runs those tests on the same bytes.

All fifteen were obtained from the **Crystallography Open Database**
(https://www.crystallography.net/cod/), which places its contents in the
public domain under the CC0 1.0 Universal dedication. Each file carries its
own `_cod_database_code`. They are test fixtures: nothing here is installed
with FACET, and the six structures FACET does ship are listed in
`facet/data/examples/ATTRIBUTION.md`.

The determinations are the work of their authors, cited below as each file
cites itself.

| File | Phase | COD | Determination |
|---|---|---|---|
| `1526458_Bi2O3.cif` | Bi₂O₃ | 1526458 | Ivanov, S. A., Orlov, V. G., Tellgren, R. *et al.* (2001). *Powder Diffraction* **16**, 227. |
| `1010311_Bi2O3.cif` | Bi₂O₃ | 1010311 | Sillén, L. G. (1938). *Arkiv för Kemi, Mineralogi och Geologi A* **12**, 1. |
| `2100844_Bi2O3.cif` | γ-Bi₂O₃ | 2100844 | Radaev, S. F., Simonov, V. I. & Kargin, Yu. F. (1992). *Acta Crystallographica Section B* **48**, 604. |
| `1526788_Na(BiO3).cif` | NaBiO₃ | 1526788 | Kumada, N., Kinomura, N. & Sleight, A. W. (2000). *Materials Research Bulletin* **35**, 2397. |
| `1004091_BiNa3O8P2.cif` | Na₃Bi(PO₄)₂ | 1004091 | Mairesse, G., Drache, M., Nowogrocki, G. *et al.* (1990). *Phase Transitions* **27**, 91. |
| `1519099_Na3Bi(PO4)2.cif` | Na₃Bi₅(PO₄)₆ | 1519099 | Arbib, E. H., Chaminade, J.-P., Darriet, J. *et al.* (2000). *Solid State Sciences* **2**, 243. |
| `7023719_BiPO4.cif` | BiPO₄ | 7023719 | Achary, S. N., Errandonea, D., Muñoz, A. *et al.* (2013). *Dalton Transactions* **42**, 14999. |
| `7023720_BiPO4.cif` | BiPO₄ | 7023720 | Achary, S. N., Errandonea, D., Muñoz, A. *et al.* (2013). *Dalton Transactions* **42**, 14999. |
| `2104291_BiB3O6.cif` | BiB₃O₆ | 2104291 | Dinnebier, R. E., Hinrichsen, B., Lennie, A. *et al.* (2009). *Acta Crystallographica Section B* **65**, 1. |
| `1010622_BiI3-P3.cif` | BiI₃ | 1010622 | Braekken, H. (1930). *Zeitschrift für Kristallographie* **74**, 67. |
| `1010622_BiI3-R-3H.cif` | BiI₃ | 8104287 | Trotter, J. & Zobel, T. (1966). *Zeitschrift für Kristallographie* **123**, 67. |
| `9004097_Cryolite.cif` | Cryolite, Na₃AlF₆ | 9004097 | Hawthorne, F. C. & Ferguson, R. B. (1975). *The Canadian Mineralogist* **13**, 377. |
| `4030257_iodosodalite.cif` | Iodosodalite, Na₈Al₆Si₆O₂₄I₂ | 4030257 | Weller, M. T. & Wong, G. (1989). *European Journal of Solid State and Inorganic Chemistry* **26**, 619. |
| `9011871_bismutoferrite.cif` | Bismutoferrite, BiFe₂Si₂O₈(OH) | 9011871 | Zhukhlistov, A. P. & Zvyagin, B. B. (1977). *Soviet Physics Crystallography* **22**, 419. |
| `1541071_NaSbOSiO4_Pagnoux1992.cif` | NaSbOSiO₄ | 1541071 | Pagnoux, C., Favard, J. F., Piffard, Y. *et al.* (1992). *Comptes Rendus de l'Académie des Sciences, Série 2* **314**, 151. |

`1010622_BiI3-R-3H.cif` keeps the name the tests ask for; the entry it holds
is COD 8104287, the rhombohedral determination, as the file itself states.

Two structures that some tests name are not here, because they come from a
licensed database and may not be passed on: those tests skip wherever the
files are absent. The tests that sweep a whole collection of structures skip
unless `FACET_REFERENCE_CIFS` names a copy of that collection.
