# CLTM Framework: Cross-Lingual Transfer Analysis for Speech Processing

This repository contains the code and experimental framework developed for the Master’s Thesis:

**“On the Language-Agnostic Nature of Speech Processing Tasks”**  
Pol Buitrago Esteve  
MSc in Advanced Telecommunication Technologies (MATT)  
Universitat Politècnica de Catalunya (UPC), 2026

The project introduces the **Cross-Lingual Transfer Matrix (CLTM)**, a general and task-agnostic framework for systematically analyzing cross-lingual transfer effects in speech processing.

---

## Overview

Multilingual speech models often rely on cross-lingual data transfer to compensate for data scarcity in low-resource languages. However, transfer effects are highly task- and language-dependent, and there is no unified methodology to quantify or compare these effects across tasks.

This framework addresses that gap by:

- Defining a **pairwise, quantitative measure of cross-lingual transfer**
- Applying it consistently across **heterogeneous speech tasks**
- Enabling **task-level characterization of language dependence**

The CLTM captures how adding data from a donor language affects performance on a target language under controlled experimental conditions.

---

## Tasks Covered

The framework is evaluated on three representative speech processing tasks:

- **Gender Identification** (paralinguistic)
- **Speaker Verification** (paralinguistic)
- **Automatic Speech Recognition (ASR)** (linguistic)

This combination allows direct comparison between linguistic and paralinguistic tasks in terms of language dependence.

---

## Methodology

- Backbone: **mHuBERT-147**, a massively multilingual self-supervised speech model
- Data source: **Mozilla Common Voice**
- Controlled fine-tuning with:
  - Fixed data regimes
  - Multiple random seeds
  - Deterministic training outside seeded randomness
- Automatic computation of:
  - Cross-Lingual Transfer Matrices (CLTM)
  - Aggregated and task-level transfer metrics
- Graph-based and geometric analyses for interpretability

All experiments are designed to isolate the effect of donor-language data while controlling for confounding factors such as data volume and initialization.

---

## Repository Structure

```

cltm-framework/
├── src/                         # Core source code
│   ├── CLTM/                    # Cross-Lingual Transfer Matrix definition and utilities
│   ├── gender/                  # Gender identification task pipeline
│   ├── speaker/                 # Speaker verification task pipeline
│   ├── speaker-no-validation/   # Speaker verification without validation split
│   ├── asr/                     # Automatic Speech Recognition task pipeline
│   ├── outputs/                 # Experiment outputs (metrics, logs, intermediate results)
│   └── **pycache**/             # Python cache files
│
├── scripts/                     # Experiment orchestration and analysis scripts
│   ├── data/                    # Dataset indexing and metadata utilities
│   ├── interval/                # Optimal data-interval estimation scripts
│   ├── labels/                  # Label processing and filtering utilities
│   ├── matrix/                  # CLTM computation and aggregation scripts
│   ├── results/                 # Result consolidation and post-processing
│   ├── recover/                 # Experiment recovery and checkpoint handling
│   ├── tools/                   # Shared utilities and helper scripts
│   ├── visualization/           # Plotting and transfer visualization
│   └── examine.py               # Experiment inspection and debugging
│
├── backyard/                    # Development, exploratory, and scratch experiments
└── README.md

```

> **Note:** Raw datasets (`data/`) and HuggingFace cache (`hf_cache/`) are excluded from the repository due to size and licensing constraints.

---

## Reproducibility

The framework is designed with reproducibility as a core principle:

- All experiments are seed-controlled
- Configurations are fully logged
- CLTM computation is deterministic given fixed inputs
- Results can be reproduced by rerunning the same configuration files

The experiments were executed on the **MareNostrum 5 supercomputer** at the Barcelona Supercomputing Center (BSC), using GPU clusters managed via SLURM.

---

## Citation

If you use this framework or the CLTM methodology in your research, please cite.

---

## License

This repository is intended for **research and academic use**.

---

## Contact

For questions or collaboration inquiries:

**Pol Buitrago Esteve**
GitHub: [https://github.com/Pol-Buitrago](https://github.com/Pol-Buitrago)
