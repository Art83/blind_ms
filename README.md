# MS Blind-Spot Analysis

This study answers why does shotgun mass spectrometry miss proteins that are
present in a tissue? Analysis harmonises three independent open consortia
(GTEx-MS body map proteome, Human Protein Atlas antibody/IHC data, PaxDb
abundance), determines proteins that IHC says are genuinely present, and asks which
protein properties predict an MS miss once abundance is accounted for.

## Findings

## Run order
if all necessary files are in `data/` (see more below), pipeline should be fully reproducible on any machine with python.


| # | Script                      | Stage               | Reads                                                                                                                                                                         | Writes                                                                            |
|---|-----------------------------|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| 1 | `1.1.load_gtex.py`          | Preprocessing stage | `gtex_protein_tissue_median.csv`, `gtex_protein_ts_score.csv`                                                                                                                 | `gtex_long.tsv`, `gtex_crosswalk.tsv`                                             |
| 2 | `1.2.load_ihc.py`           | Preprocessing stage | `normal_ihc_data.tsv`                                                                                                                                                         | `gene_dict.tsv`, `ihc_celltype_long.tsv`, `ihc_tissue.tsv`, `ihc_level_audit.tsv` |
| 3 | `1.3.load_paxdb.py`         | Preprocessing stage | `paxdb/*.txt`, `paxdb/hs_whole_body.txt`, `gene_dict.tsv`                                                                                                                     | `paxdb_long.tsv`, `paxdb_wholebody.tsv`                                           |
| 4 | `2.build_grid.py`           | Preprocessing stage | `gene_dict.tsv`, `gtex_long.tsv`, `ihc_tissue.tsv`, `paxdb_long.tsv`, `paxdb_wholebody.tsv`                                                                                   | `grid.tsv`, `grid_class_counts.tsv`, `grid_qc.txt`                                |
| 5 | `3.1.feature_transcript.py` | Building features   | `ensembl/mart_export_features.txt`, `ensembl/mart_export_fasta.txt`                                                                                                           | `features_transcript.tsv`                                                         |
| 6 | `3.2.feature_annotation.py` | Building features   | `corum_human_complexes.txt`, `Summary_Counts.default_predictions.txt`, `Gene_info.txt`, `sorfs_human_5utr.csv`, `gene_coordinates_grch38.tsv`, `eclip_beds/`, `gene_dict.tsv` | `features_annotation.tsv`                                                         |
| 7 | `3.3.feature_biophysics.py` | Building features   | `uniprot_human.tsv`, `gene_dict.tsv`, `features_transcript.tsv`, `features_annotation.tsv`                                                                                    | `features_protein.tsv`                                                            |
| 8 | `3.4.feature_structure.py`  | Building features   | `alphafold/UP000005640_9606_HUMAN_v*.tar`, `half_life_protein.csv`, `features_protein.tsv`, `gene_dict.tsv`                                                                   | `features_structure.tsv`, `features_structure.txt`                                |




## Data provenance 
All dataset are in open access and can be downloaded without any user agreements. The only exception is iPSC datasets.
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

