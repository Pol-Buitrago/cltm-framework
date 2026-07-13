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

* Defining a **pairwise, quantitative measure of cross-lingual transfer**
* Applying it consistently across **heterogeneous speech tasks**
* Enabling **task-level characterization of language dependence**

The CLTM captures how adding data from a donor language affects performance on a target language under controlled experimental conditions.

---

## Tasks Covered

The framework is evaluated on three representative speech processing tasks:

* **Gender Identification** (paralinguistic)
* **Speaker Verification** (paralinguistic)
* **Automatic Speech Recognition (ASR)** (linguistic)

This combination allows direct comparison between linguistic and paralinguistic tasks in terms of language dependence.

---

## Architectures Used

### Gender Identification

* **mHuBERT-147** (src/HuBERT/gender): massively multilingual self-supervised speech model, fine-tuned for gender classification.

### Speaker Verification

1. **mHuBERT-147 (SID-based embeddings)** (src/HuBERT/speaker & speaker-no-validation)
   Pretrained HuBERT encoder fine-tuned with a speaker identification objective. Embeddings are L2-normalized; classification head discarded after training.

2. **ECAPA-TDNN** (src/ECAPA)
   Time-delay neural network generating fixed-dimensional speaker embeddings from acoustic features. Optimized with AAM-Softmax and cross-entropy loss.

3. **Siamese Network** (src/siamese)
   Learns speaker embeddings via pairwise similarity optimization using contrastive loss. Shared-weight network processes pairs of utterances.

4. **SpeechLLM** (src/speechLLM)
   Large pretrained speech model generates embeddings; pairwise representations fed into a lightweight MLP classifier with sigmoid cross-entropy to predict same-speaker probability.

### Automatic Speech Recognition (ASR)

* **mHuBERT-147** (src/HuBERT/asr): fine-tuned for transcription across multiple languages using Mozilla Common Voice datasets.

---

## CLTM Matrices

The CLTM matrices illustrate **cross-lingual transfer effects** for all tasks and architectures. All images are included below for immediate reference.

### Gender Identification (mHuBERT)

![CLTM Gender mHuBERT](./imgs/gender_mhubert.png)

### Speaker Verification

* **mHuBERT (SID-based embeddings)**
  ![CLTM SV mHuBERT](./imgs/sv_mhubert.png)

* **ECAPA-TDNN**
  ![CLTM SV ECAPA-TDNN](./imgs/sv_ecapa_tdnn.png)

* **Siamese Network**
  ![CLTM SV Siamese](./imgs/sv_siamese.png)

* **SpeechLLM**
  ![CLTM SV SpeechLLM](./imgs/sv_speechllm.png)

### ASR (mHuBERT)

![CLTM ASR mHuBERT](./imgs/asr_mhubert.png)

> All matrices are automatically generated from experimental outputs in `src/*/outputs/` and are ready for direct inspection or inclusion in publications.

---

## Methodology

* Controlled fine-tuning with:

  * Fixed data regimes
  * Multiple random seeds
  * Deterministic training outside seeded randomness
* Automatic computation of:

  * Cross-Lingual Transfer Matrices (CLTM)
  * Aggregated and task-level transfer metrics
* Graph-based and geometric analyses for interpretability
* Experiments isolate the effect of donor-language data while controlling for confounding factors such as data volume and initialization

---

## Repository Structure

```
cltm-framework/
├── src/
│   ├── CLTM/                 # CLTM computation, utilities, figures
│   ├── HuBERT/               # mHuBERT pipelines (asr, gender, speaker, speaker-no-validation)
│   ├── ECAPA/                # ECAPA-TDNN speaker verification pipeline
│   ├── siamese/              # Siamese network speaker verification pipeline
│   ├── speechLLM/            # SpeechLLM speaker verification pipeline
│   ├── outputs/              # Experiment outputs (metrics, logs, matrices)
│   └── __pycache__/
├── scripts/
│   ├── data/
│   ├── examine.py
│   ├── interval/
│   ├── labels/
│   ├── results/
│   ├── tools/
├── backyard/                  # Development, exploratory, and scratch experiments
├── data/                      # Raw datasets (excluded from repo)
├── hf_cache/                  # HuggingFace cache (excluded from repo)
└── README.md
```

> **Note:** Raw datasets (`data/`) and HuggingFace cache (`hf_cache/`) are excluded due to size and licensing constraints.

---

## Reproducibility

* All experiments are seed-controlled
* Configurations are fully logged
* CLTM computation is deterministic given fixed inputs
* Results can be reproduced by rerunning the same configuration files

Experiments were executed on the **MareNostrum 5 supercomputer** at the Barcelona Supercomputing Center (BSC), using GPU clusters managed via SLURM.

---

## Citation

If you use this framework or the Cross-Lingual Transfer Matrix (CLTM) methodology in your research, please cite the original paper:

```bibtex
@article{buitrago2026cltransfer,
  title={Quantifying Cross-Lingual Transfer in Paralinguistic Speech Tasks},
  author={Buitrago, Pol and Pareras, Oriol and Costa, Federico and Hernando, Javier},
  journal={arXiv preprint arXiv:2603.08231},
  year={2026},
  note={Accepted at Interspeech 2026}
}
```

This repository contains the reference implementation of the CLTM framework introduced in the above publication.

---

## License

This project is licensed under the **Creative Commons Attribution 4.0 International License** (CC BY 4.0).

Proper attribution is required:
**Pol Buitrago Esteve** – [https://github.com/Pol-Buitrago](https://github.com/Pol-Buitrago)

[Official license page](https://creativecommons.org/licenses/by/4.0/)

---

## Related Publications

The CLTM framework has been introduced and subsequently extended through the following publications.

#### 1. CLTM methodology

Introduces the Cross-Lingual Transfer Matrix (CLTM), a general framework for quantifying cross-lingual transfer in speech processing, and validates it on two paralinguistic tasks: **Gender Identification** and **Speaker Verification**.

```bibtex
@article{buitrago2026cltransfer,
  title={Quantifying Cross-Lingual Transfer in Paralinguistic Speech Tasks},
  author={Buitrago, Pol and Pareras, Oriol and Costa, Federico and Hernando, Javier},
  journal={arXiv preprint arXiv:2603.08231},
  year={2026},
  note={Accepted at Interspeech 2026}
}
```

#### 2. Speaker Verification architectures

Applies the CLTM framework to compare the cross-lingual transfer properties of multiple speaker verification architectures.

```bibtex
@inproceedings{buitrago2026towards,
  title={Towards Language-Agnostic Speaker Verification: A Cross-Lingual Transfer Study of Architectures},
  author={Buitrago, Pol and Hernando, Javier},
  booktitle={Proceedings of Odyssey 2026: The Speaker and Language Recognition Workshop},
  pages={298--305},
  year={2026},
  doi={10.21437/Odyssey.2026-44}
}
```

#### 3. Speaker and language disentanglement

Evaluates the impact of speaker variability on cross-lingual speaker verification performance within the Iberian languages setting, using the CLTM framework to disentangle speaker and language effects.

```bibtex
@article{buitrago2026disentangling,
  title={Disentangling Speaker and Language Effects in Cross-Lingual Speaker Verification for Iberian Languages},
  author={Buitrago, Pol and Hernando, Javier},
  journal={arXiv preprint arXiv:2607.01161},
  year={2026}
}
```

## Contact

**Pol Buitrago Esteve**
GitHub: [https://github.com/Pol-Buitrago](https://github.com/Pol-Buitrago)

---
