#region Imports
from argsparser import ArgsParser
import pandas as pd
import numpy as np
import json
import os
import torch
import jiwer
from datetime import datetime
import logging
import copy

from datasets import load_dataset, DatasetDict, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, TrainingArguments
from transformers import Trainer, AutoFeatureExtractor, AutoConfig, AutoProcessor, AutoModelForCTC
from transformers import HubertForCTC, set_seed
from accelerate import Accelerator

from dataclasses import dataclass
from typing import Any, Dict, List, Union

from utils.utils import normalize_text
from utils.utils_logging import LoggerHelper
from utils.collators import DataCollatorCTCWithPadding
#endregion



class ASRFineTuner(LoggerHelper):
    def __init__(self, params):
        self.params = params

        self.set_accelerator()  # crea self.accelerator

        # inicializar LoggerHelper con el accelerator
        super().__init__(self.accelerator)

        self.start_datetime = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
        self.get_pre_trained_model_name()
        self.get_pre_trained_model_family()
        self.set_fetures_key_name()

        self.ft_model_output_dir = self.params.fine_tuned_output_folder
        os.environ["HF_DATASETS_CACHE"] = os.path.join(self.ft_model_output_dir, "hf_datasets_cache")
        self.set_log_file_handler()
        if self.params.set_seed: 
            self.set_seed()
        
        self.log_on_main(
            "We are assuming that every training audio is mono and with 16K sampling rate!",
            type="warning"
        )
        self.log_on_main(f"Going to use pre trained model from: {self.params.pre_trained_hf_model}")
        self.log_on_main(f"Going to fine-tune a {self.pre_trained_model_name} model for ASR.")
        self.log_on_main(f"Going to save fine-tuned model files to: {self.ft_model_output_dir}")

    
    def set_accelerator(self):
        self.accelerator = Accelerator()


    def set_log_file_handler(self, logger_level = "info"):
        """
        Set a logging file handler.
        """
        if self.accelerator.is_local_main_process:
        
            if not os.path.exists(self.ft_model_output_dir):
                os.makedirs(self.ft_model_output_dir)
            
            logger_file_name = f"{self.start_datetime}_logs.txt"
            logger_file_name = logger_file_name.replace(':', '_').replace(' ', '_').replace('-', '_')

            logger_file_path = os.path.join(self.ft_model_output_dir, logger_file_name)
            logger_file_handler = logging.FileHandler(logger_file_path, mode = 'w')
            
            if logger_level == "info":
                logger_file_handler.setLevel(logging.INFO) 
            else:
                logger_file_handler.setLevel(logging.DEBUG)

            logger_file_handler.setFormatter(self.logger_formatter)

            logging.getLogger("utils.utils_logging").addHandler(logger_file_handler)

    
    def get_pre_trained_model_name(self):
        self.log_on_main("Extracting pre_trained_model_name...")
        path = str(self.params.pre_trained_hf_model).rstrip('/')
        parts = path.split('/')
        lower_parts = [p.lower() for p in parts]
        if "models" in lower_parts:
            idx = lower_parts.index("models")
            if idx + 1 < len(parts):
                self.pre_trained_model_name = parts[idx + 1]
            else:
                self.pre_trained_model_name = os.path.basename(path)
        else:
            self.pre_trained_model_name = os.path.basename(path)
        self.log_on_main(f"pre_trained_model_name: {self.pre_trained_model_name}")
    
    
    def get_pre_trained_model_family(self):

        name = self.pre_trained_model_name.lower()

        if "hubert" in name:
            self.model_family = "hubert"
        else:
            raise Exception(
                f"There is not a family defined for the model {self.pre_trained_model_name}"
            )
        self.log_on_main(f"model_family: {self.model_family}")

    
    def set_fetures_key_name(self):
        """
        For w2v-bert models, the corresponding feature_extractor generates features with a key name 
        that is different than the other models.
        We define this key name to use it along the process.
        """

        if self.model_family == "w2v_bert":
            self.features_key_name = "input_features"
        else:
            self.features_key_name = "input_values"

        self.log_on_main(f"features_key_name: {self.features_key_name}")

    
    def set_seed(self):
        self.seed = int(self.params.seed)
        self.log_on_main(f"Setting seed for experiments reproducibility (seed={self.seed})...")
        set_seed(self.seed)


    def create_dataset(self, dataloader_folder_dir, load_test_split):
        """
        Create train, validation, and optionally test datasets using a local loading script.
        Simplified for Common Voice: no origin detection or splitting by origin is performed.
        """

        self.log_on_main("Entered create_dataset...")

        # Change working directory to dataloader_folder_dir for loading_script.py to work
        original_wd = os.getcwd()
        os.chdir(dataloader_folder_dir)
        self.log_on_main(f"Working directory: {os.getcwd()}", type="debug")

        # Load train and validation splits
        train_dataset = load_dataset(
            f"{dataloader_folder_dir}/loading_script.py",
            trust_remote_code=True,
            split="train",
        )
        self.log_on_main("Train dataset created!")

        validation_dataset = load_dataset(
            f"{dataloader_folder_dir}/loading_script.py",
            trust_remote_code=True,
            split="validation",
        )
        self.log_on_main("Validation dataset created!")

        dataset_dict = {
            "train": train_dataset,
            "validation": validation_dataset,
        }

        if load_test_split:
            # Load test split (no extra mapping)
            test_dataset = load_dataset(
                f"{dataloader_folder_dir}/loading_script.py",
                trust_remote_code=True,
                split="test",
            )

            # Add the test dataset directly under the key "test"
            dataset_dict["test"] = test_dataset
            self.log_on_main("Test dataset added as 'test' (no origin mapping).")

        # Wrap all datasets in a DatasetDict
        dataset = DatasetDict(dataset_dict)

        # Restore original working directory
        os.chdir(original_wd)
        self.log_on_main(f"Working directory: {os.getcwd()}", type="debug")
        self.log_on_main(f"Dataset structure: {dataset}")
        self.log_on_main("Exited create_dataset...")

        return dataset

    
    # Data Preprocessing Recommended by HuggingFace
    def prepare_text_map_function(self, batch):

        """Function to preprocess the dataset with the .map method"""
        transcription = batch["text"]

        transcription = normalize_text(transcription)

        batch["text"] = transcription

        return batch

    
    def prepare_dataset(self, do_text_preparation = True):
        self.log_on_main("Entered prepare_dataset...")

        # Columns refactor
        self.dataset = self.dataset.remove_columns(["audio_id", 'language'])
        self.dataset = self.dataset.rename_columns({"normalized_text": "text"})

        # Text preparation
        if do_text_preparation:
            self.dataset = self.dataset.map(
                self.prepare_text_map_function, 
                num_proc=4,
                )
            
            self.log_on_main("Searching for forbidden sentences...")
            for split in self.dataset.keys():
                self.log_on_main(f"Split: {split}")
                forbidden_sentences_count =  sum([1 if text == "FORBIDDEN_SENTENCE" else 0 for text in self.dataset[split]["text"]])
                if forbidden_sentences_count > 0: 
                    forbidden_sentences_paths =  [path if text == "FORBIDDEN_SENTENCE" else 0 for text, path in zip(self.dataset[split]["text"], self.dataset[split]["audio"])]
                    self.log_on_main(f"forbidden_sentences_paths: {forbidden_sentences_paths}")
                assert forbidden_sentences_count == 0, f"There should be 0 ASR labels with the text 'FORBIDDEN_SENTENCE' (instead of {forbidden_sentences_count})!"
            self.log_on_main("No forbidden sentences, everything ok!")

        # Audio preparation
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=16_000))

        self.log_on_main("Exited prepare_dataset...")
    
    
    def create_vocab(self):
        self.log_on_main("Entered create_vocab...")

        if self.accelerator.is_local_main_process:
            # extraer textos ya normalizados desde los splits
            train_texts = self.dataset["train"]["text"]
            val_texts   = self.dataset["validation"]["text"]

            all_text = " ".join(train_texts) + " " + " ".join(val_texts)
            # ya debería estar normalizado por prepare_dataset, pero forzamos idempotencia:
            all_text = normalize_text(all_text)  # si es muy largo, puedes normalizar por fragmentos

            # construir conjunto de caracteres válidos (sin espacios)
            chars = set(ch for ch in all_text if ch != " ")
            vocab_list = sorted(chars)

            # reservar tokens especiales (orden fijo)
            # Reservamos PAD=0, UNK=1, WORD_SEP='|'=2 (si quieres)
            # ajusta indices según convención; aquí hacemos PAD=0, UNK=1, '|'=2, y luego chars desde 3
            self.vocab_dict = {
                "<PAD>": 0,
                "<UNK>": 1,
                "|": 2,
            }
            offset = len(self.vocab_dict)
            for k, ch in enumerate(vocab_list):
                self.vocab_dict[ch] = k + offset

            # sanity checks
            if "<S>" in self.vocab_dict or "</S>" in self.vocab_dict:
                raise RuntimeError("BOS/EOS tokens found in vocab, they must not be used for CTC.")

            self.log_on_main(f"Created vocab (len={len(self.vocab_dict)}): sample: {dict(list(self.vocab_dict.items())[:30])}")

            self.log_on_main(f"Saving vocab into: {self.params.vocab_file_path}")
            with open(self.params.vocab_file_path, 'w', encoding='utf-8') as vocab_file:
                json.dump(self.vocab_dict, vocab_file, ensure_ascii=False)
            self.log_on_main(f"Saved! vocab len: {len(self.vocab_dict)}")

        self.accelerator.wait_for_everyone()
        self.log_on_main("Exited create_vocab...")


    def load_tokenizer(self):
        self.log_on_main("Entered load_tokenizer...")

        self.tokenizer = Wav2Vec2CTCTokenizer(
            vocab_file=self.params.vocab_file_path,
            unk_token="<UNK>",
            pad_token="<PAD>",
            word_delimiter_token="|",
            bos_token=None,
            eos_token=None,
            do_lower_case=False,  # seguro junto a normalize_text
        )

        # logging como antes...
        self.log_on_main(f"self.tokenizer.get_vocab(): {self.tokenizer.get_vocab()}")

        # test de tokenización realista, sin tags ni uppercase forzado
        self.log_on_main("Testing the tokenization with some phrase...")
        phrase = "abcdef ghijk lmnop qrstuvwx yz àá çèé íñ óú ü 1234567890"
        self.log_on_main(f"phrase: {phrase}")
        tokenized_phrase = self.tokenizer._tokenize(phrase)
        self.log_on_main(f"tokenized_phrase: {tokenized_phrase}")
        decoded_phrase = self.tokenizer.decode([self.tokenizer._convert_token_to_id(token) for token in tokenized_phrase])
        self.log_on_main(f"decoded_phrase: {decoded_phrase}")
        self.log_on_main("Testing the tokenization with some phrase done!")
        
        self.log_on_main("Exited load_tokenizer...")
    
    
    def load_feature_extractor(self):

        self.log_on_main("Entered load_feature_extractor...")

        """
        AutoFeatureExtractor.from_pretrained: This is a generic feature extractor class that will be 
        instantiated as one of the feature extractor classes of the library when created.
        The feature extractor class to instantiate is selected based on the model_type property of the 
        config object (either passed as an argument or loaded from pretrained_model_name_or_path if possible), 
        or when it’s missing, by falling back to using pattern matching on pretrained_model_name_or_path

        Wav2Vec2 models that have set `config.feat_extract_norm == "group"`, such as hubert, 
            have **not** been trained using `attention_mask`. 
            For such models, `input_values` should simply be padded with 0 and no `attention_mask`
            should be passed.
        For Wav2Vec2 models that have set config.feat_extract_norm == "layer", such as wav2vec2-lv60, 
        attention_mask should be passed for batched inference.

        Normalizing can help to significantly
            improve the performance for some models

        source code: https://github.com/huggingface/transformers/blob/10feacd88aef9569e240b7e3833ab32b297e4460/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py#L31
        """

        # This is going to load config values from the file preprocessor_config.json in the repo
        
        if self.model_family == "w2v_bert":
            self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
                self.params.pre_trained_hf_model,
            )
        else:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.params.pre_trained_hf_model,
                local_files_only=True,
            )

        if self.params.feature_extractor_do_normalize == 'true':
            self.feature_extractor.do_normalize = True
        elif self.params.feature_extractor_do_normalize == 'false':
            self.feature_extractor.do_normalize = False
        else:
            self.log_on_main(f"Using original configuration for self.feature_extractor.do_normalize")
        
        self.log_on_main(f"self.feature_extractor: {self.feature_extractor}")
        
        self.log_on_main("Exited load_feature_extractor...")
    
    
    def create_processor(self):

        self.log_on_main("Entered create_processor...")

        if self.model_family == "w2v_bert":
            self.processor = Wav2Vec2BertProcessor(
                feature_extractor=self.feature_extractor, 
                tokenizer=self.tokenizer,
                )
        else:
            self.processor = Wav2Vec2Processor(
                feature_extractor=self.feature_extractor, 
                tokenizer=self.tokenizer,
            )

        self.log_on_main(f"self.processor: {self.processor}")

        self.log_on_main("Exited create_processor...")
    
    
    def process_dataset_map_function(self, batch):
        # Usamos siempre 'audio' (hemos dejado la columna así en prepare_dataset)
        audio = batch.get("audio", None)
        if audio is None or "array" not in audio:
            raise RuntimeError(f"Audio missing or invalid for example {batch.get('audio_id','unknown')}")

        arr = audio["array"]
        sr = int(audio["sampling_rate"])

        # If multichannel, downmix to mono (mean) — más seguro que tomar solo el primer canal
        if getattr(arr, "ndim", 1) == 2:
            # arr shape: (n_samples, n_channels) -> mean across channels
            arr = np.mean(arr, axis=1)

        # Ensure float32
        arr = arr.astype(np.float32)

        # Llamada al processor — sampling_rate viene del cast_column (debería ser 16000)
        batch_processed = self.processor(
            arr,
            sampling_rate=sr,
            text=batch["text"],
        )

        # compute input length from the features key
        batch_processed["input_length"] = len(batch_processed[self.features_key_name][0])

        return batch_processed

    
    def process_dataset(self):
        self.log_on_main("Entered process_dataset...")
        
        self.dataset = self.dataset.map(
            self.process_dataset_map_function, 
            remove_columns=self.dataset.column_names["train"], 
            num_proc=4,
        )

        self.log_on_main(f"dataset['train'][0]: {self.dataset['train'][0]}", type = "debug")

        self.log_on_main("Exited process_dataset...")


    def create_data_collator(self):
        self.data_collator = DataCollatorCTCWithPadding(
            processor=self.processor, 
            padding="longest",
            features_key_name = self.features_key_name,
        )
    
    
    def load_model(self):
        self.log_on_main("Entered load_model")

        if self.model_family == "hubert":
            config = AutoConfig.from_pretrained(
                self.params.pre_trained_hf_model,
                local_files_only=True,
            )

            config.vocab_size = len(self.tokenizer.get_vocab())
            config.pad_token_id = self.processor.tokenizer.pad_token_id

            vocab_size = len(self.tokenizer.get_vocab())
            assert vocab_size == self.tokenizer.vocab_size or True, "Comprobar vocab_size"
            self.log_on_main(f"Using vocab_size={vocab_size}, pad_token_id={self.processor.tokenizer.pad_token_id}")

            self.model = HubertForCTC.from_pretrained(
                self.params.pre_trained_hf_model,
                config=config,
            )
        else:
            raise Exception(f"There is not a CTC class defined for the model {self.pre_trained_model_name}")

        self.log_on_main(f"self.model.target_lang: {self.model.target_lang}")
        self.log_on_main(f"self.model.lm_head: {self.model.lm_head}", type = "debug")
        
        self.log_on_main(f"self.model.config.vocab_size: {self.model.config.vocab_size}")

        self.log_on_main(f"self.model: \n {self.model}")

        self.log_on_main("Exited load_model")
    

    def freeze_parameters(self):
        self.log_on_main(f"Freezing some model parameters...")

        if self.params.model_freezing == "base_model":
            self.log_on_main("Freezed base model!")
            self.model.freeze_base_model()
        elif self.params.model_freezing == "feature_encoder":
            self.log_on_main("Freezed feature encoder!")
            self.model.freeze_feature_encoder()
        else:
            self.log_on_main("No freezing")

        #for index, (name, parameter) in enumerate(self.model.named_parameters()):
        #    if name not in ["lm_head.weight", "lm_head.bias"]:
        #        parameter.requires_grad = False

        for index, (name, parameter) in enumerate(self.model.named_parameters()):
            self.log_on_main(f"{name} {parameter.numel()}, trainable: {parameter.requires_grad}", type = "debug")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params  = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log_on_main(f"Model total params: {total_params}")
        self.log_on_main(f"Model total trainable params: {trainable_params} ({trainable_params/total_params*100:.2f}%)")

        self.log_on_main(f"Freezing some model parameters done!")

    
    def compute_metrics(self, pred, compute_cer = False, compute_wers = False, compute_cers = False):
        #self.log_on_main("Entered compute_metrics")

        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id
        pred_str = self.processor.batch_decode(pred_ids)
        
        # we do not want to group tokens when computing the metrics
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

        wers = [jiwer.wer(r, h) for r, h in zip(label_str, pred_str)]
        wer = float(np.mean(wers))

        if compute_cer: 
            cer = jiwer.cer(reference = label_str, hypothesis = pred_str)        
        # Compute more detailed wer statistics (maybe a little computational consuming)
        if compute_wers: 
            wers = [jiwer.wer(reference = ref, hypothesis = hyp) for (ref, hyp) in zip(label_str, pred_str)]
        if compute_cers: 
            cers = [jiwer.cer(reference = ref, hypothesis = hyp) for (ref, hyp) in zip(label_str, pred_str)]

        results = {
            "wer": wer,
        }

        if compute_cer: results["cer"] = cer
        if compute_wers: results["wers"] = wers
        if compute_cers: results["cers"] = cers
        
        return results 
    

    def set_training_args(self):
        self.training_args = TrainingArguments(
            disable_tqdm=False,
            eval_strategy = "epoch", # "epoch" or "steps"
            fp16 = True, # False -> 32
            group_by_length = True,
            length_column_name="input_length",
            greater_is_better = False,
            logging_nan_inf_filter = False,
            load_best_model_at_end = True, 
            logging_strategy = "steps",
            logging_steps = 50,  # how often to log 
            max_grad_norm = 1.0,
            metric_for_best_model = "eval_wer",
            per_device_train_batch_size = 8, #8
            report_to=["tensorboard"], #"none", "wandb", "tensorboard"
            save_safetensors = False,
            save_only_model = True,
            save_total_limit = 2,
            save_strategy = "epoch",
            #
            # custom params
            output_dir = self.ft_model_output_dir,
            learning_rate = self.params.learning_rate,
            weight_decay = self.params.weight_decay,
            lr_scheduler_type = self.params.lr_scheduler_type,
            num_train_epochs = self.params.num_train_epochs,
            warmup_ratio = self.params.warmup_ratio,
            #
            # experimental arguments
            #run_name=f"run_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",  # name of the W&B run (optional)
            #full_determinism=True,
            #do_train = True,
            #auto_find_batch_size = True,
            log_level = "info",
            eval_accumulation_steps = self.params.eval_accumulation_steps, # needed for big models
            per_device_eval_batch_size = self.params.per_device_eval_batch_size, # needed for big models
            gradient_accumulation_steps = 1,
            gradient_checkpointing = False,
        )

        if self.params.set_seed:
            self.training_args.seed=self.seed
            self.training_args.data_seed=self.seed

        self.log_on_main(f"self.training_args: \n {self.training_args}")

    
    def train(self):
        self.log_on_main("Preparing Trainer with validation-only evaluation...")

        # Build Trainer: evaluation only on validation split during training
        self.trainer = self.accelerator.prepare(
            Trainer(
                model=self.model,
                data_collator=self.data_collator,
                args=self.training_args,
                compute_metrics=self.compute_metrics,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["validation"],
                processing_class=self.processor,
            )
        )

        try:
            self.log_on_main("Starting training...")
            if self.params.resume_from_checkpoint:
                self.log_on_main(f"Resuming training from {self.params.resume_from_checkpoint}...")
                self.trainer.train(resume_from_checkpoint=self.params.resume_from_checkpoint)
            else:
                self.trainer.train()
            self.log_on_main("Training done!")
        except Exception as e:
            self.log_on_main(f"Failed to finish training! Error: {repr(e)}", type="warning")

    
    def final_evaluation(self):
        """
        Very small final evaluation: use the Trainer prepared during training to evaluate the 'test' split.
        Assumes self.trainer exists (train() was executed before). Does not teardown distributed groups.
        """
        self.log_on_main("Starting final evaluation (simple) ...")

        # Quick skip if no test split
        if "test" not in self.dataset:
            self.log_on_main("No 'test' split present, skipping final evaluation.", type="warning")
            return pd.DataFrame(columns=["dataset", "wer"])

        # Ensure trainer exists
        if not hasattr(self, "trainer") or self.trainer is None:
            raise RuntimeError("final_evaluation requires self.trainer to exist (run train() first in this process).")

        # Run evaluation exactly as validation does
        self.log_on_main("Running evaluate() on TEST split (using prepared Trainer)...")
        test_metrics = self.trainer.evaluate(eval_dataset=self.dataset["test"])
        self.log_on_main(f"Test metrics: {test_metrics}")

        # Build a small summary dataframe (wer first)
        rows = []
        wer_key_candidates = [k for k in test_metrics.keys() if "wer" in k]
        wer_key = wer_key_candidates[0] if wer_key_candidates else "wer"
        rows.append({"dataset": "test", "wer": test_metrics.get(wer_key, None), **test_metrics})

        df_full_evaluation_metrics = pd.DataFrame(rows)
        df_full_evaluation_metrics = df_full_evaluation_metrics.sort_values("wer", ascending=True)

        # Save only from local main process
        if self.accelerator.is_local_main_process:
            os.makedirs(self.ft_model_output_dir, exist_ok=True)
            out_path = os.path.join(self.ft_model_output_dir, "df_full_evaluation_metrics.csv")
            df_full_evaluation_metrics.to_csv(out_path, sep="\t", index=False)
            self.log_on_main(f"Saved df_full_evaluation_metrics to {out_path}")

        self.log_on_main("Final evaluation (simple) done!")
        return df_full_evaluation_metrics


    def main(self):
        self.dataset = self.create_dataset(self.params.dataloader_folder_dir, load_test_split = True)
        self.prepare_dataset(do_text_preparation = True) # this text preparation was already done when creating the ASR labels
        # verificar que no quedan mayúsculas
        for split in ["train", "validation"]:
            texts = self.dataset[split]["text"]
            bad = any(any(ch.isupper() for ch in t if ch.isalpha()) for t in texts)
            if bad:
                raise RuntimeError("Transcripciones contienen mayúsculas tras normalize_text. Revisa normalize_text.")

        if self.params.create_vocab_file: self.create_vocab()
        self.load_tokenizer()
        self.load_feature_extractor()
        self.create_processor()
        self.process_dataset()
        self.create_data_collator()
        self.load_model()
        self.freeze_parameters()
        self.set_training_args()
        self.train()
        self.final_evaluation()
        self.dump_log_history(self.trainer)
        self.accelerator.end_training()


if __name__=="__main__":

    args_parser = ArgsParser()
    args_parser.main()
    parameters = args_parser.arguments

    self = ASRFineTuner(parameters)
    self.main()