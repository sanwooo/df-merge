from collections import defaultdict
from transformers import AutoTokenizer, T5TokenizerFast
from datasets import load_from_disk, DatasetDict, concatenate_datasets
from promptsource.promptsource.templates import DatasetTemplates, Template


class DatasetReader:
    """
        Dataset Reader object reads datasets and has attributes specific to the dataset.
    """

    def __init__(self, dataset_path, template_stash, tokenizer: T5TokenizerFast, template_idx: int=0):
        
        self.dataset_path = dataset_path
        self.template_stash = template_stash
        self.template_idx = template_idx # template index to choose
        self.tokenizer = tokenizer

        self.template = self._get_dataset_template(None, None)
        self.label_to_index = dict()

    def _read_orig_dataset(self):
        dataset = load_from_disk(self.dataset_path)
        return dataset
    
    def _get_dataset_template(self, template_names_to_ignore: list[str], metrics_to_use: list[str]) -> Template:
        """
            Returns valid template.
            choose the first one among the valid ones.
        """
        valid_templates = []

        for template in DatasetTemplates(*self.template_stash).templates.values():
            # Filter out templates that
            # 1) are not designed for original task
            # 2) have different metrics than we want to use
            # 3) are ones that we want to ignore based on the name
            if template.metadata.original_task:
                ignore_template = False

                for metric in template.metadata.metrics:
                    if metric not in metrics_to_use:
                        ignore_template = True 
                    
                for template_name in template_names_to_ignore:
                    if template_name == template.name:
                        ignore_template = True
                
                if not ignore_template:
                    valid_templates.append(template)
        
        selected_template = valid_templates[self.template_idx]
        return selected_template
        
    def _preprocess_train(self, example):
        x, y = self.template.apply(example)
        inputs = self.tokenizer(x, truncation=True)
        labels = self.tokenizer(text_target=y, truncation=True)

        model_inputs = {
            "input_ids" : inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"],
        }
        return model_inputs
    
    def _preprocess_eval(self, example):
        x, y = self.template.apply(example)
        answer_choices = self.template.get_answer_choices_list(example)
        answer_index = answer_choices.index(y)
        inputs = self.tokenizer(x, truncation=True)
        labels_all_choices = self.tokenizer(text_target=answer_choices)

        model_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels_all_choices": labels_all_choices["input_ids"],
            "answer_index": answer_index,
        }
        return model_inputs

    def get_dataset(self, is_train: bool):
        orig_dataset = self._read_orig_dataset()
        dataset_dict = dict()
        if is_train:
            for split in ['train', 'validation']:
                dataset_dict[split] = orig_dataset[split].map(self._preprocess_train, batched=False, \
                                                              remove_columns=[x for x in orig_dataset[split].column_names], \
                                                              load_from_cache_file=False, keep_in_memory=True)
        else:
            for split in ['validation', 'test']:
                dataset_dict[split] = orig_dataset[split].map(self._preprocess_eval, batched=False, \
                                                              remove_columns=[x for x in orig_dataset[split].column_names], \
                                                              load_from_cache_file=False, keep_in_memory=True)

        return DatasetDict(dataset_dict)

class PAWSReader(DatasetReader):
    def __init__(self, tokenizer: T5TokenizerFast):
        super().__init__(
            dataset_path = "./dataset/paws/labeled_final",
            template_stash = ("paws", "labeled_final"),
            tokenizer = tokenizer,
        )

    def _get_dataset_template(self, template_names_to_ignore, metrics_to_use):
        return super()._get_dataset_template([], ["Accuracy"])

class QASCReader(DatasetReader):
    def __init__(self, tokenizer: T5TokenizerFast):
        super().__init__(
            dataset_path = "./dataset/qasc",
            template_stash = ("qasc",),
            tokenizer = tokenizer,
        )

    def _read_orig_dataset(self):
        orig_dataset = super()._read_orig_dataset()
        valid_test = orig_dataset["validation"].train_test_split(test_size=0.5, shuffle=False)
        return DatasetDict({
            'train': orig_dataset['train'],
            'validation': valid_test['train'],
            'test': valid_test['test'],
        })  

    def _get_dataset_template(self, template_names_to_ignore, metrics_to_use):
        return super()._get_dataset_template([], ["Accuracy"])

class QuaRTzReader(DatasetReader):
    def __init__(self, tokenizer: T5TokenizerFast):
        super().__init__(
            dataset_path = "./dataset/quartz",
            template_stash = ("quartz",),
            tokenizer = tokenizer,
        )
    
    def _get_dataset_template(self, template_names_to_ignore, metrics_to_use):
        return super()._get_dataset_template([], ["Accuracy"])

class StoryClozeReader(DatasetReader):
    def __init__(self, tokenizer: T5TokenizerFast):
        super().__init__(
            dataset_path = "./dataset/story_cloze/2016/dataset",
            template_stash = ("story_cloze", "2016"),
            tokenizer = tokenizer,
        )
    
    def _read_orig_dataset(self):
        orig_dataset = super()._read_orig_dataset()
        valid_test = orig_dataset["test"].train_test_split(test_size=0.5, shuffle=False)
        return DatasetDict({
            'train': orig_dataset['validation'],
            'validation': valid_test['train'],
            'test': valid_test['test'],
        })  
    
    def _get_dataset_template(self, template_names_to_ignore, metrics_to_use):
        return super()._get_dataset_template([], ["Accuracy"])

class WikiQAReader(DatasetReader):
    def __init__(self, tokenizer: T5TokenizerFast):
        super().__init__(
            dataset_path = "./dataset/wiki_qa",
            template_stash = ("wiki_qa",),
            tokenizer = tokenizer,
        )
    
    def _get_dataset_template(self, template_names_to_ignore, metrics_to_use):
        return super()._get_dataset_template([], ["Accuracy"])

class WinograndeReader(DatasetReader):
    def __init__(self, tokenizer: T5TokenizerFast):
        super().__init__(
            dataset_path = "./dataset/winogrande/winogrande_xl",
            template_stash = ("winogrande", "winogrande_xl"),
            tokenizer = tokenizer,
        )
    
    def _read_orig_dataset(self):
        orig_dataset = super()._read_orig_dataset()
        valid_test = orig_dataset["validation"].train_test_split(test_size=0.5, shuffle=False)
        return DatasetDict({
            'train': orig_dataset['train'],
            'validation': valid_test['train'],
            'test': valid_test['test'],
        })  
    
    def _get_dataset_template(self, template_names_to_ignore, metrics_to_use):
        return super()._get_dataset_template([], ["Accuracy"])


DATASET_CLASSES = {
    'paws': PAWSReader,
    'qasc': QASCReader,
    'quartz': QuaRTzReader,
    'story_cloze': StoryClozeReader,
    'wiki_qa': WikiQAReader,
    'winogrande': WinograndeReader,
}

class MixtureReader:
    def __init__(self, tokenizer: T5TokenizerFast):
        """
            mixture reader for multi-task learning.
        """
        self.dataset_readers = [DATASET_CLASSES[x](tokenizer) for x in DATASET_CLASSES.keys()]
    
    def get_dataset(self, is_train: bool):
        assert is_train==True , "mixture is not applicable for evaluation."
        mixture_dataset_dict = defaultdict(list)
        for dataset_reader in self.dataset_readers:
            dataset_dict = dataset_reader.get_dataset(is_train)
            for split in dataset_dict:
                mixture_dataset_dict[split].append(dataset_dict[split])

        for split in mixture_dataset_dict:
            mixture_dataset_dict[split] = concatenate_datasets(mixture_dataset_dict[split])
        
        return DatasetDict(mixture_dataset_dict)
    
def get_dataset_reader(dataset_name: str, tokenizer: T5TokenizerFast) -> DatasetReader:
    if dataset_name == 'mixture':
        return MixtureReader(tokenizer)
    return DATASET_CLASSES[dataset_name](tokenizer)