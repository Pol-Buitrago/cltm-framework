import argparse

class ArgsParser:

    def __init__(self):
        
        self.initialize_parser()

    
    def initialize_parser(self):

        self.parser = argparse.ArgumentParser(
            description = 'Script for fine-tune a SSL HF audio model (like HuBERT or wav2vec2) to ASR.',
            )


    def add_parser_args(self):

        self.parser.add_argument(
            '--dataloader_folder_dir',
            type = str, 
            help = "This folder must contain the laoding_script.py and a data folder \
                with train.tsv, dev.tsv and test.tsv."
            )
        
        self.parser.add_argument(
            '--fine_tuned_output_folder',
            type = str, 
            help = "Folder to dump the fine-tuned model files. The script will automatically create \
                a subfolder with the model name + 'ft_for_asr' suffix and inside that, \
                    a subfolder with the starting training datetime as unique identifier."
            )

        self.parser.add_argument(
            '--pre_trained_hf_model',
            type = str, 
            help = "HuggingFace dowloaded pre-trained model folder path to fine-tune. \
                This folder must have all the HF repo files."
            )

        self.parser.add_argument(
            '--resume_from_checkpoint',
            type = str, 
            help = "HuggingFace checkpoint to resume training from."
            ) 

        self.parser.add_argument(
            '--chars_to_ignore_regex',
            type = str, 
            default = '[\,\?\.\!\-\;\:\"]',
            help = "Remove these chars when processing transcripts."
            ) 
        
        self.parser.add_argument(
            '--create_vocab_file',
            action=argparse.BooleanOptionalAction,
            help = "If True, a vocabulary is going to be created based on the train and dev data."
            )
        
        self.parser.add_argument(
            '--vocab_file_path',
            type = str,
            help = "JSON file path to load the vocabulary (and also to save it if create_vocab_file is True)."
            ) 

        self.parser.add_argument(
            '--learning_rate',
            type = float, 
            default = 0.0001,
            ) 

        self.parser.add_argument(
            '--lr_scheduler_type',
            type = str, 
            help = "Several HF options: linear, constant, polynomial, etc",
            default = "linear",
            ) 

        self.parser.add_argument(
            '--weight_decay',
            type = float, 
            default = 0.001,
            ) 
            
        self.parser.add_argument(
            '--num_train_epochs',
            type = int, 
            default = 5,
            ) 

        self.parser.add_argument(
            '--warmup_ratio',
            type = float, 
            default = 0.2,
            )

        self.parser.add_argument(
            '--eval_accumulation_steps',
            type = int, 
            )

        self.parser.add_argument(
            '--per_device_eval_batch_size',
            type = int, 
            )
        
        self.parser.add_argument(
            '--set_seed',
            action=argparse.BooleanOptionalAction,
            help = "Set to True for experiments reproducibility."
            )
            
        self.parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for reproducibility"
        )

        self.parser.add_argument(
            '--model_freezing',
            type=str,
            default="feature_encoder",
            choices=["no-freezing", "base_model", "feature_encoder"],
        )

        self.parser.add_argument(
            '--feature_extractor_do_normalize',
            type=str,
            default="original_config",
            choices=["original_config", "true", "false"],
            help = "Feature extractor normalization option."
            )

        self.parser.add_argument(
            '--trained_model_to_evaluate_path',
            type = str,
            help = "Checkpoint path of the trained ASR model to evaluate. \
                This is only used for the evaluation script, not the fine-tunning."
            )
            

    def main(self):

        self.add_parser_args()
        self.arguments = self.parser.parse_args()
