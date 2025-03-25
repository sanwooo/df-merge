# Dynamic Fisher-weighted Model Merging via Bayesian Optimization

This is the code implementation to reproduce __DF-Merge__ of the NAACL 2025 paper 
"Dynamic Fisher-weighted Model Merging via Bayesian Optimization" by Sanwoo Lee, Jiahao Liu, Qifan Wang, Jingang Wang, Xunliang Cai and Yunfang Wu*.

## Dependencies

To run the code, please install all dependencies specified in `pyproject.toml` via poetry. Install [poetry](https://python-poetry.org/docs/) and run the following command.
```
poetry install
poetry shell
```

## Downloads

Since we ran the code in an offline environment, datasets and models should be downloaded beforehand.

Create `.dataset` folder and download six datasets (paws, qasc, quartz, story_cloze, wiki_qa and winogrande) under this directory.
```python
import os
from datasets import load_dataset

root_path = ".dataset"
for ds_stash in [("paws", "labeled_final"), ("qasc",), ("quartz",), ("story_cloze", "2016", "dataset"), ("wiki_qa",), ("winogrande", "winogrande_xl"))]:
    if ds_stash[0] == "story_cloze":
        ds = load_dataset(*ds_stash, data_dir=ds_stash[2], trust_remote_code=True)
    else:
        ds = load_dataset(*ds_stash, trust_remote_code=True)
    ds.save_to_disk(os.path.join(root_path, *ds_stash))
```

Create `.models` folder and download [t5-base](https://huggingface.co/google-t5/t5-base) and [t5-large](https://huggingface.co/google-t5/t5-large)
under this folder.

Additionally, download [promptsource](https://github.com/bigscience-workshop/promptsource) package and install.
```sh
git clone https://github.com/bigscience-workshop/promptsource.git
cd promptsource
pip install -e .
cd ../
```
## fine-tune models and save checkpoints

The module [src/finetune.py](src/finetune.py) trains t5-base/t5-large model on single task or mixture of all tasks.
To train on a single task, run:
```sh
$dataset = paws # [paws, qasc, quartz, story_cloze, wiki_qa, winogrande] 
$model_name_or_path = t5-base # [t5-base, t5-large]
$config_path = ./configs/t5-single-task.yaml
$experiment_name = default #any name for your experiment
$seed = 42
python src/finetune.py --dataset $dataset --model_name_or_path $model_name_or_path --config_path $config_path --experiment_name $experiment_name --seed $seed 
```
This will save model checkpoint at directory `./experiments/finetune/$experiment_name/ckpt/$model_name_or_path/$dataset/seed$seed/`.

(optional, this is not required to run DF-Merge) To train on the mixture of all tasks (multi-task), run:
```
$dataset = mixture
$model_name_or_path = t5-base # [t5-base, t5-large]
$config_path = ./configs/t5-multi-task.yaml
$experiment_name = train # any name for your experiment
$seed = 42
$early_stopping_patience = 20

python src/finetune.py --dataset $dataset --model_name_or_path $model_name_or_path --config_path $config_path --experiment_name $experiment_name --early_stopping_patience $early_stopping_patience --seed $seed 
```
This will save model checkpoint at directory `./experiments/finetune/$experiment_name/ckpt/$model_name_or_path/mixture/seed$seed/`.

## run DF-Merge and evaluate performance
Having run `src/finetune.py` on each of the six datasets, you can run df-merge and evaluate performance via `src/df-merge.py`.

```sh
$ckpt_root = experiments/finetune/train/ckpt 
$model_name_or_path = t5-base # [t5-base, t5-large]
$seed = 42
$acquisition_fn = ei # [ei, ucb]
$experiment_name = merge # any name for your experiment

python src/df-merge.py --ckpt_root $ckpt_root --model_name_or_path $model_name_or_path --seed $seed --acquisition_fn $acquisition_fn --experiment_name $experiment_name 
```
> [!IMPORTANT]
> Please make sure that `$ckpt_root`, `$model_name_or_path`, `$seed` and `$experiment_name used for src/finetune.py`(`train` in this example)  match the checkpoint saved by running `src/finetune.py`.

This will save the evaluation result (the merged model is not saved) at directory `experiments/df-merge/$experiment_name/metrics/$model_name_or_path/$acquisition_fn/use_fisher_True/seed$seed.`

## (optional) inference

If you wish to evaluate the performance of each fine-tuned or multi-task model, please refer to [src/inference.py](src/inference.py).

## Acknowledgement
This codebase references the codebase below and we thank their efforts.
```bibtext
@inproceedings{
      yadav2023tiesmerging,
      title={{TIES}-Merging: Resolving Interference When Merging Models},
      author={Prateek Yadav and Derek Tam and Leshem Choshen and Colin Raffel and Mohit Bansal},
      booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
      year={2023},
      url={https://openreview.net/forum?id=xtaX3WyCj1}
}
```



