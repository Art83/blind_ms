# MS Blind-Spot Analysis

This study asks why does shotgun mass spectrometry miss proteins that are
present in a tissue. Analysis harmonises three independent open consortia
(GTEx-MS body map proteome, Human Protein Atlas antibody/IHC data, PaxDb
abundance) and determines proteins that IHC says are genuinely present, and asks which
protein properties predict an MS miss once abundance is accounted for.

Outcomes:
- **measured**: MS found the protein in this tissue. 64,960 rows.(MS-bright)
- **measured_absent**: MS found the protein somewhere in the body map, in at least one other tissue, but not in other(s). 6,603 rows. The protein is technically detectable by MS but something about context(tissue) causes the miss.
- **not_measured**: MS never found the protein in any tissue in the whole study. 17,480 rows. The protein is a miss everywhere.

Why two MS-dark groups instead of just one pooled:

A protein that MS never finds anywhere might be low everywhere, or its sequence has properties (few tryptic peptides, hydrophobic) that don't agree with MS. Thus, analysing **measured vs not_measured** is mainly chemistry/biophysics test.
In contrast, A protein that MS finds in liver but misses in lung cannot be missed just because of its properties/function. Thus, analysing **measured vs measured_absent** is mainly about "dilution", where protein is low in that tissue (present in only a few cell types) and gets swamped when the whole organ is homogenised.

## Findings
- Abundance dominates (beta=2.16), tryptic peptide yield second (beta=0.69, p=4e-15). Transmembrane count, hydrophobicity and size are null once yield is in. Yield survives the tissue-matched abundance covariate (+0.35, p 5e-4), IHC thresholds, peptide windows and reliability floors. In-sample AUC 0.90.
- Dilution with cell count weighting is null across six tissues (p=0.84) and marginal in the two tissues with more than two scored cell types (beta=0.085, p=0.04). The main conclusion though is that there's not enough resolution. Single-cell proteomics needs to happen to increase granularity.
- Yield stays between +0.40 and +0.70 across IHC thresholds, peptide windows and antibody reliability floors while TM count stays null.
- In-silico tryptic count against observed peptides (PeptideAtlas), Spearman +0.40 raw, +0.30 after controlling abundance and length, and +0.27 using the size-normalised version (peptides per kDa). So more cleavable means more peptides detected, independent of how big or how abundant the protein is.
- Gene-level GTEx detection fraction against PeptideAtlas peptide count, Spearman +0.67. Proteins GTEx never finds in any tissue have a median of 21 PeptideAtlas peptides against 132 for proteins GTEx finds somewhere. Number that matters for re-framing of this project: only 3.4% of GTEx-dark proteins have never been seen in PeptideAtlas. So dark means harder to detect, not invisible.
- **ML** Three nested-CV gradient-boosting models, one row per protein: **intrinsic** (sequence, structure, localisation, complexes, **no abundance, no translation block**) AUC 0.79, 
  **practical** (intrinsic plus translation-regulation features, every input computable from the gene) AUC 0.83, 
  **+abundance** AUC 0.89. In the practical model the translation block is the top group by a wide margin, so its gain over intrinsic is mostly
  abundance predicted from the gene. Proteotypicity is the only group that keeps importance once abundance is known; transmembrane, signal peptide
  and PTM groups contribute nothing. Within the middle abundance third, detection rises from 76% to 95% across tryptic-yield quartiles, and the
  lowest 5% of intrinsic scores are 74% MS-dark (3.1x base rate).
- **Per-tissue score.** Same three models on gene x tissue rows: **intrinsic AUC 0.78, practical 0.81, +abundance 0.90**, flat across all ten tissues, and
  raw abundance alone reaches 0.865. The model separates detected proteins
  from proteins missed in every tissue (0.91) better than from proteins
  missed only in the tissue at hand (0.87), because a single-tissue miss is
  mostly a low-abundance protein near the detection limit. Gene-level collapse reproduces the previous stage AUCs.
- **Missing proteome.** A sequence-only scorer (AUC 0.74 vs 0.79 for the full intrinsic model) applied to all 20,431 reviewed human proteins gives
  PE2-4 a median detectability of 0.36 against 0.78 for PE1, with the gap holding in every length band, and 15% of PE2-4 have no unique tryptic
  peptide (1.4% of PE1). The 90 proteins promoted to PE1 between UniProt exports scored 0.65 while those still missing scored 0.36 (p 1e-13)
- **Protease switch.** In six cell lines digested with six proteases (PXD024364), 88% of GTEx-dark genes seen in the data were found with
  trypsin anyway; of the 183 that trypsin still missed, 31% reached two peptides under an alternative enzyme. Rescue tracked low in-silico tryptic
  yield (p=6e-4), not abundance (p 0.41) or membrane status (p 0.34)
- 
## Run order
if all necessary files are in `data/` (see more below the details of each dataset), pipeline should be fully reproducible on any machine with python.


| #  | Script                           | Stage               | Reads                                                                                                                                                                                                      | Writes                                                                            |
|----|----------------------------------|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| 1  | `1.1.load_gtex.py`               | Preprocessing stage | `gtex_protein_tissue_median.csv`, `gtex_protein_ts_score.csv`                                                                                                                                              | `gtex_long.tsv`, `gtex_crosswalk.tsv`                                             |
| 2  | `1.2.load_ihc.py`                | Preprocessing stage | `normal_ihc_data.tsv`                                                                                                                                                                                      | `gene_dict.tsv`, `ihc_celltype_long.tsv`, `ihc_tissue.tsv`, `ihc_level_audit.tsv` |
| 3  | `1.3.load_paxdb.py`              | Preprocessing stage | `paxdb/*.txt`, `paxdb/hs_whole_body.txt`, `gene_dict.tsv`                                                                                                                                                  | `paxdb_long.tsv`, `paxdb_wholebody.tsv`                                           |
| 4  | `2.build_grid.py`                | Preprocessing stage | `gene_dict.tsv`, `gtex_long.tsv`, `ihc_tissue.tsv`, `paxdb_long.tsv`, `paxdb_wholebody.tsv`                                                                                                                | `grid.tsv`, `grid_class_counts.tsv`, `grid_qc.txt`                                |
| 5  | `3.1.feature_transcript.py`      | Building features   | `ensembl/mart_export_features.txt`, `ensembl/mart_export_fasta.txt`                                                                                                                                        | `features_transcript.tsv`                                                         |
| 6  | `3.2.feature_annotation.py`      | Building features   | `corum_human_complexes.txt`, `Summary_Counts.default_predictions.txt`, `Gene_info.txt`, `sorfs_human_5utr.csv`, `gene_coordinates_grch38.tsv`, `eclip_beds/`, `gene_dict.tsv`                              | `features_annotation.tsv`                                                         |
| 7  | `3.3.feature_biophysics.py`      | Building features   | `uniprot_human.tsv`, `gene_dict.tsv`, `features_transcript.tsv`, `features_annotation.tsv`                                                                                                                 | `features_protein.tsv`                                                            |
| 8  | `3.4.feature_structure.py`       | Building features   | `alphafold/UP000005640_9606_HUMAN_v*.tar`, `half_life_protein.csv`, `features_protein.tsv`, `gene_dict.tsv`                                                                                                | `features_structure.tsv`, `features_structure.txt`                                |
| 9  | `3.5.feature_peptides.py`        | Building features   | `uniprot_human.tsv`, `features_protein.tsv` (check only)                                                                                                                                                   | `features_peptides.tsv`                                                           |
| 10 | `4.model_msdark.py`              | Analysis            | `grid.tsv`, `features_protein.tsv`                                                                                                                                                                         | `model_table.tsv`, `model_summary.txt`                                            |
| 11 | `dilution_rescue.py`             | Diagnostics         | `celltype_proportions.csv`, `consensus_bridge_map.csv`, `ihc_celltype_long.tsv`, `model_table.tsv`                                                                                                         | `dilution_covariate.tsv`, `dilution_unbridged.tsv`, `dilution_summary.txt`        |
| 12 | `robustness.py`                  | Diagnostics         | `model_table.tsv`, `ihc_celltype_long.tsv`, `uniprot_human.tsv`, `gene_dict.tsv`                                                                                                                           | `robustness_summary.txt`                                                          |
| 13 | `peptideatlas_validate.py`       | Diagnostics         | `APD_ensembl_hits.tsv`, `ensp2ensg.txt`, `features_protein.tsv`, `model_table.tsv`                                                                                                                         | `peptideatlas_per_gene.tsv`, `peptideatlas_validation.txt`                        |
| 14 | `5.1.ml_detectability.py`        | Analysis            | `features_protein.tsv`, `features_peptides.tsv`, `features_structure.tsv`, `model_table.tsv`, `peptideatlas_per_gene.tsv`, `paxdb_wholebody.tsv`                                                           | `ml_summary.txt`, `ml_per_gene_predictions.tsv`                                   |
| 15 | `label_noise_floor.py`           | Diagnostics         | `gtex_protein_tissue_median.csv`, `grid.tsv`, `peptideatlas_per_gene.tsv`, `ml_per_gene_predictions.tsv`                                                                                                   | `label_noise_subsites.tsv`, `label_noise_subsites.tsv`                            |
| 16 | `5.2.ml_tissue_detectability.py` | Analysis            | `model_table.tsv`, `features_protein.tsv`, `features_peptides.tsv`, `label_noise_subsites.tsv`                                                                                                             | `tissue_detectability_predictions.tsv`, `tissue_detectability.txt`                |
| 17 | `6.proteome_census.py`           | Analysis            | `uniprot_human.tsv` (Protein existence), `uniprotkb_pe.tsv` (previous release, for the promotion check), `features_protein.tsv`, `features_peptides.tsv`, `model_table.tsv`, `ml_per_gene_predictions.tsv` | `proteome_census.tsv`, `proteome_census.txt`                                      |
| 16 | `multiprotease_rescue.py`        | Diagnostics         |  `proteinGroups.txt`, `peptides.txt` (PXD024364 MaxQuant txt), `ensp2ensg.txt`, `uniprot_human.tsv`, `model_table.tsv`, `features_protein.tsv`, `gene_dict.tsv`                                            | `multiprotease_rescue.tsv`                                                        |

## Translation
| # | Script                    | Reads | Writes |
|---|---------------------------|-------|--------|
| 1 | `7.organoid_platforms.py` | `organoids/somalogic/*medNormSMP*.adat`, `organoids/olink/olink_slim.csv`, `organoids/MS/Human_report.pg_matrix.tsv`, `organoids/gen_meta.csv`, `organoids/MS_meta.csv`, `features_protein.tsv`, `gene_dict.tsv`, `uniprot_human.tsv` | `organoid_long.tsv`, `organoid_features.tsv`, `organoid_platforms.txt` |


## Data provenance 
All datasets are in open access and can be downloaded without any user agreements. The only exception is iPSC data.
- **GTEx-MS**: Jiang et al. 2020, Cell, PXD016999.
- **HPA IHC**: `normal_ihc_data.tsv`, HPA release 25.0
- **PaxDb**: integrated datasets, publication_year 2025 per file header. Per-tissue input weights are in each file's `#weights` line and summarised in `paxdb_provenance.tsv`
- **PeptideAtlas**: human build 502 (2021), 2,322 samples
- **UniProt**: exported via website, September 2026, 20,431 entries, 21 columns. 
  Every protein-level feature in the paper is computed from this file on the canonical mature
  chain
- **Ensembl**: release 116 (June 2026 archive BioMart, features export and
  canonical protein-coding cDNA FASTA, `data/ensembl/`), for the transcript
  block only
- **CORUM**: 5.0, coreComplexes
- **TargetScan**: 7.2, human summary counts and gene info
- **sORFs.org**: 5'UTR sORF export, accessed 2024
- **ENCODE eCLIP**: narrowPeak BEDs, GRCh38, downloaded 2026-09-08
- **AlphaFold DB**: human proteome UP000005640, model v6, tar downloaded 2026-09-08
- **Protein half-life**: Mathieson et al. 2018, Nat Commun, human primary cells (B cells, NK cells, hepatocytes, monocytes).
- **PXD024364**: Sinitcyn et al. 2023, MassIVE MSV000086944, `search/txt.zip`
- **Organoids**: DIA-NN version ____, SomaScan 11k v5.0 (ADATs dated 2026-01-26 media, 2026-02-11 organoids), Olink Explore HT (NPX Map 2.0.0, NovaSeq X Plus)

