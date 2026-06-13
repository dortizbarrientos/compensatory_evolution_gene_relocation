# Data provenance

Raw published data underlying the empirical confrontation in `docs/model.pdf`
(Part III). All files are third-party supplementary material, included here for
reproducibility; cite the original papers.

## Drosophila — Schrider et al. (2011), Genome Research 21:2087–2095
- `Schrider_SuppTable1.xls` — polymorphic retroCNVs (parental gene, coordinates,
  lines carrying each). The 34 retroCNVs; 4 originate on the X
  (sgg, Mur2B, CG11160, CG2662), 30 on autosomes.
- `Schrider_SuppTable2.xls` — intron-deletion polymorphisms (not used in the model test).
- `Genome_Res_-2011-Schrider-2087-95.pdf` — the paper. Table 3 (poly vs fixed,
  X- vs autosome-origin) is the source of the Drosophila row in `tab:datatest`.

## Human — Schrider et al. (2013), PLOS Genetics 9:e1003242
- `pgen_1003242_s004.xls` (S1) — discovery genomes.
- `pgen_1003242_s005.xls` (S2) — retroCNV coordinates + insertion sites.
- `pgen_1003242_s006.xls` (S3) — validation genomes.
- `pgen_1003242_s007.xls` (S4) — retroCNVs/genomes per analysis.
- `pgen_1003242_s008.xls` (S5) — movements, X- vs autosome-origin (located).
- `pgen_1003242_s009.xls` (S6) — movements, to-X vs to-autosome (destination).
- `pgen_1003242_s010.xls` (S7) — origin including unknown insertion site.
- `pgen_1003242_s011.xls` (S8) — trio genotypes.
- `shrider_humans.pdf` — the paper. Table 1 / Tables S5–S7 are the source of the
  human rows in `tab:datatest`.

## Note on how these enter the analysis
`src/retrogene_data_test.py` uses the contingency counts (X- vs autosome-origin,
polymorphic vs fixed) extracted from the tables above; the counts are documented
inline in that script. The raw tables are kept here so the extraction can be
re-checked or extended (e.g. deriving the Drosophila counts directly from
`Schrider_SuppTable1.xls` by parsing parental coordinates).

## Not yet present (needed for the next step — the silencer test)
The against-the-grain gene list needs **sex-biased / male-germline expression**
(e.g. FlyAtlas, testis scRNA-seq) to classify relocations, plus the
fixed-relocation polarizations (Bai et al. 2007; Vibranovski et al. 2009).
These are not in this folder; see the data-gathering protocol.
