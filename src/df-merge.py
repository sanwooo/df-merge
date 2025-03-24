import os
import sys
PROJECT_ROOT = "."
MODEL_ROOT = "./models" 
sys.path.append(PROJECT_ROOT)
import argparse
import copy
from collections import OrderedDict
from datetime import datetime
from shutil import copytree, rmtree

import numpy as np
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer, set_seed
from datasets import Dataset
from bayes_opt import BayesianOptimization, acquisition
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from src.utils.logger import logger, log_info_file
from src.data.dataset_readers import get_dataset_reader
from src.model.vector import Vector
from src.model.utils import load_checkpoints
from src.eval.evaluator import MetricEvaluator, FisherEvaluator


def setup_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", type=str, default=['paws', 'qasc', 'quartz', 'story_cloze', 'wiki_qa', 'winogrande'])
    parser.add_argument("--ckpt_root", type=str,)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--seed", type=int, help="random seed for reproducibility")
    parser.add_argument("--acquisition_fn", type=str, choices=["ei", "ucb"])
    parser.add_argument("--use_fisher", action="store_true", default=False)
    parser.add_argument("--approximate_by_sampling", action="store_true", default=False)
    parser.add_argument("--init_points", type=int, default=10)
    parser.add_argument("--n_iter", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--experiment_name", type=str, default=None)
    
    args = parser.parse_args()

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

    args.bayes_opt_ckpt_path = os.path.join(
        experiment_dir,
        'ckpt',
        args.model_name_or_path,
        args.acquisition_fn,
        f"use_fisher_{args.use_fisher}",
        f"seed{args.seed}",
        "log.log",
    )
    os.makedirs(os.path.dirname(args.bayes_opt_ckpt_path), exist_ok=True)
    args.metrics_path = os.path.join(
        experiment_dir,
        'metrics',
        args.model_name_or_path,
        args.acquisition_fn,
        f"use_fisher_{args.use_fisher}",
        f"seed{args.seed}",
        "metrics.json",
    )
    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    log_info_file(os.path.join(experiment_dir, "merge.log"))
    logger.info(vars(args)) 
    return args

class DFMerge:

    def __init__(
            self,
            pt_model: T5ForConditionalGeneration,
            ft_model_dict: dict[str, T5ForConditionalGeneration],
            tokenizer: AutoTokenizer,
            ds_dict: dict[str, Dataset],
            use_fisher: bool = True
    ):
        """
            Dynamic Fisher-weighted Model Merge.
        """
        self.base_model = copy.deepcopy(pt_model)
        self.pt_vector = Vector(pt_model=pt_model)
        self.ft_vector_dict = {ds: Vector(pt_model=pt_model, ft_model=ft_model) for ds, ft_model in ft_model_dict.items()}
        self.tokenizer = tokenizer
        self.ds_dict = ds_dict
        self.fisher_evaluator = None
        self.metric_evaluator = None
        self.use_fisher = use_fisher

    def _merge_step(
            self,
            coef_dict: dict[str, float],
    ):
        M = len(coef_dict)
        f, fwv = 0, None
        for dataset, coef in coef_dict.items():
            # linear interpolation
            lerp_vector = self.pt_vector + coef * self.ft_vector_dict[dataset]
            lerp_model = lerp_vector.apply_to(self.base_model)
            if self.use_fisher:
                current_fisher = self.fisher_evaluator.compute_diagonal_fisher(lerp_model, self.tokenizer, self.ds_dict[dataset], dataset)
            else:
                current_fisher = 1
            f += current_fisher
            fwv += current_fisher * coef * self.ft_vector_dict[dataset]
        
        del current_fisher
        merged_vector = ( ( (M * fwv) / f ) ) + self.pt_vector
        merged_model = merged_vector.apply_to(self.base_model)
        return merged_model

    def _black_box_function(
            self,
            coef_dict: dict[str, float],
    ) -> float:
        merged_model = self._merge_step(coef_dict)
        acc_dict = {}
        for dataset_name, dataset in self.ds_dict.items():
            metrics = self.metric_evaluator.evaluate(merged_model, self.tokenizer, dataset)
            acc_dict[dataset_name] = metrics['accuracy']
        avg_acc = np.mean(list(acc_dict.values()))
        if self.optimizer.max is None or avg_acc > self.optimizer.max['target']:
            self.fisher_evaluator.best_indicies_dict = copy.deepcopy(self.fisher_evaluator.indicies_dict)

        logger.info(f"avg_acc: {avg_acc}, acc_dict: {acc_dict}")
        
        return avg_acc
    
    def merge(
            self,
            acquisition_fn: str,
            seed: int,
            init_points: int,
            n_iter: int,
            num_samples: int,
            approximate_by_sampling: bool,
            eval_batch_size: int,
            log_path: str,
    ) -> T5ForConditionalGeneration:
        
        self.fisher_evaluator = FisherEvaluator(num_samples, approximate_by_sampling=approximate_by_sampling)
        self.metric_evaluator = MetricEvaluator(split='validation', eval_batch_size=eval_batch_size)

        pbounds = {ds: (0, 1) for ds in self.ds_dict.keys()}
        self.optimizer = BayesianOptimization(
            f=lambda **coef_dict: self._black_box_function(
                coef_dict,
            ),
            pbounds=pbounds,
            acquisition_function=acquisition_fn,
            random_state=seed,
            verbose=2,
        )
        bayes_opt_logger = JSONLogger(path=log_path)
        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, bayes_opt_logger)
        self.optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )
        optimal_coef = self.optimizer.max['params']
        optimal_merged_model = self._merge_step(optimal_coef)
        return optimal_merged_model
            

if __name__ == '__main__':

    args = setup_experiment()   
    set_seed(args.seed)
    pt_model, ft_model_dict, tokenizer = load_checkpoints(MODEL_ROOT, args.ckpt_root, args.model_name_or_path, args.seed, args.datasets)
    dataset_dict = {ds: get_dataset_reader(ds, tokenizer).get_dataset(is_train=False) for ds in args.datasets}
    merge_wrapper = DFMerge(
        pt_model=pt_model,
        ft_model_dict=ft_model_dict,
        tokenizer=tokenizer,
        ds_dict=dataset_dict,
        use_fisher= args.use_fisher,
    )
    if args.acquisition_fn == 'ei':
        acquisition_fn = acquisition.ExpectedImprovement(xi=0.01)
    elif args.acquisition_fn == 'ucb':
        acquisition_fn = acquisition.UpperConfidenceBound(kappa=2.576)


    merged_model = merge_wrapper.merge(
        acquisition_fn,
        seed=args.seed,
        init_points=args.init_points,
        n_iter=args.n_iter,
        num_samples=args.num_samples,
        approximate_by_sampling=args.approximate_by_sampling,
        eval_batch_size=args.eval_batch_size,
        log_path=args.bayes_opt_ckpt_path,
    )
    # final test after merge
    test_evaluator = MetricEvaluator(split='test', eval_batch_size=args.eval_batch_size)
    for dataset_name, dataset in dataset_dict.items():
        metrics = test_evaluator.evaluate(merged_model, tokenizer, dataset)
        test_evaluator.save_metrics(metrics, os.path.join(os.path.dirname(args.metrics_path), f'{dataset_name}_metrics.json'))

