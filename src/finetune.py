import os
import sys
PROJECT_ROOT = "."
MODEL_ROOT = "./models" 
sys.path.append(PROJECT_ROOT)
import argparse
import re
from datetime import datetime
from shutil import copytree, rmtree
from transformers import HfArgumentParser, T5ForConditionalGeneration, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback
from src.utils.logger import logger, log_info_file
from src.data.dataset_readers import get_dataset_reader

def delete_subdirs_with_regex(dir_path, regex):
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        # Check if the item is a directory and matches the regex pattern
        if os.path.isdir(item_path) and re.match(regex, item):
            logger.info(f"Deleting directory: {item_path}")
            rmtree(item_path)  # Recursively delete the directory

def setup_experiment():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["paws", "qasc", "quartz", "story_cloze", "wiki_qa", "winogrande", "mixture"], default="paws")
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--seed", type=int, help="random seed for reproducibility")
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    args = parser.parse_args()

    hf_parser = HfArgumentParser((Seq2SeqTrainingArguments,))
    train_args = hf_parser.parse_yaml_file(args.config_path)[0]
    train_args.seed = args.seed
    
    # create unique experiment directory
    experiment_dir = os.path.join(PROJECT_ROOT, "experiments")
    # filename + run_name
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    if not args.experiment_name:
        now = datetime.now()
        timestamp = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second
        )
        args.experiment_name = timestamp
    experiment_dir = os.path.join(experiment_dir, file_name, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # save current source code and configuration in the experiment directory. 
    for backup_dirname in ["src", "configs"]:
        copytree(
            os.path.join(PROJECT_ROOT, backup_dirname),
            os.path.join(experiment_dir, backup_dirname),
            dirs_exist_ok=True,
        )

    # update checkpoint directory (output_dir)
    train_args.output_dir = os.path.join(
        experiment_dir,
        train_args.output_dir,
        args.model_name_or_path,
        args.dataset,
        f"seed{args.seed}",
    )

    
    return args, train_args

if __name__ == '__main__':

    args, train_args = setup_experiment()

    model = T5ForConditionalGeneration.from_pretrained(os.path.join(MODEL_ROOT, args.model_name_or_path)).cuda()
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_ROOT, args.model_name_or_path))
    
    dataset_reader = get_dataset_reader(args.dataset, tokenizer)
    dataset = dataset_reader.get_dataset(is_train=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if args.early_stopping_patience else [] 

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model()
    trainer.save_state()
    delete_subdirs_with_regex(train_args.output_dir, r"checkpoint-\d+")
