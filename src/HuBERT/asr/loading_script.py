from collections import defaultdict
import os
import csv
import datasets
from utils.utils import normalize_text
print("### LOADING SCRIPT IMPORTED ###")
print("CV_TRAIN_TSV at import:", os.environ.get("CV_TRAIN_TSV"))


_NAME = "cv_corpus_pt_asr"
_VERSION = "1.0.0"
_DESCRIPTION = "CV-Corpus 22.0 ASR subset"

_HOMEPAGE = "https://commonvoice.mozilla.org/"
_LICENSE = "CC0"
_CITATION = "Common Voice Dataset, Mozilla Foundation"

# Read paths from environment
_METADATA_TRAIN = os.environ.get("CV_TRAIN_TSV")
_METADATA_DEV = os.environ.get("CV_DEV_TSV")
_METADATA_TEST = os.environ.get("CV_TEST_TSV")
_LANGUAGE = os.environ.get("CV_LANG", "unk")

if _METADATA_TRAIN is None or _METADATA_DEV is None:
    raise RuntimeError(
        "Environment variables CV_TRAIN_TSV and CV_DEV_TSV must be defined"
    )

class CvCorpusPtConfig(datasets.BuilderConfig):
    def __init__(self, name=_NAME, **kwargs):
        super().__init__(name=name, **kwargs)


class CvCorpusPt(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version(_VERSION)
    BUILDER_CONFIGS = [CvCorpusPtConfig()]

    def _info(self):
        features = datasets.Features(
            {
                "audio_id": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=16000),
                "language": datasets.Value("string"),
                "normalized_text": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        splits = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"tsv_path": _METADATA_TRAIN},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"tsv_path": _METADATA_DEV},
            ),
        ]

        if _METADATA_TEST is not None and os.path.exists(_METADATA_TEST):
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"tsv_path": _METADATA_TEST},
                )
            )

        return splits

    def _generate_examples(self, tsv_path):
        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for i, row in enumerate(reader):
                audio_id = os.path.splitext(os.path.basename(row["audio"]))[0]
                normalized_sentence = normalize_text(row["sentence"])
                yield audio_id, {
                    "audio_id": audio_id,
                    "audio": {"path": row["audio"]},
                    "language": "pt",
                    "normalized_text": normalized_sentence,
                }

