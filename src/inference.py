import os
import sys
PROJECT_ROOT = "."
MODEL_ROOT = "./models" 
sys.path.append(PROJECT_ROOT)
import argparse
from transformers import T5ForConditionalGeneration, AutoTokenizer
from src.utils.logger import logger, log_info_file
from src.data.dataset_readers import get_dataset_reader
from src.eval.evaluator import MetricEvaluator

def setup_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--dataset", type=str, choices=["paws", "qasc", "quartz", "story_cloze", "wiki_qa", "winogrande"], default="paws")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--eval_batch_size", type=int, default=256)
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = setup_experiment()

    model = T5ForConditionalGeneration.from_pretrained(args.ckpt_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)

    dataset_reader = get_dataset_reader(args.dataset, tokenizer)
    dataset = dataset_reader.get_dataset(is_train=False)

    evaluator = MetricEvaluator(split=args.split, eval_batch_size=args.eval_batch_size)
    metrics = evaluator.evaluate(model, tokenizer, dataset)
    evaluator.save_metrics(metrics, os.path.join(args.ckpt_path, f'{args.dataset}_metrics.json'))


    