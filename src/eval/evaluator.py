import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import DataCollatorForSeq2Seq, T5ForConditionalGeneration, AutoTokenizer
from datasets import Dataset

from src.utils.logger import logger, log_info_file
from src.model.vector import Vector

class EvaluatorBase:

    def __init__(self, split: str, eval_batch_size: int):
        
        self.split = split
        self.eval_batch_Size = eval_batch_size

    def _nested_collate_fn(self, batch: list[dict], data_collator: DataCollatorForSeq2Seq):
        """
            a collate function that handles all answer choices for each datapoint
        """
        # assume number of label choices is consistent across the whole dataset
        num_choices = len(batch[0]['labels_all_choices'])
        non_label_keys = [key for key in batch[0].keys() if key != 'labels_all_choices']
        broadcast_batch = []
        for instance in batch:
            for labels in instance['labels_all_choices']:
                squeezed_instance = {key: instance[key] for key in non_label_keys}
                squeezed_instance.update({'labels': labels})
                broadcast_batch.append(squeezed_instance)
        
        pad_batch = data_collator(broadcast_batch)
        new_batch = {key : torch.stack([pad_batch[key][i] for i in range(0, len(batch)*num_choices, num_choices)]) \
                     for key in non_label_keys}
        new_batch.update({'labels': pad_batch['labels']})
        # new_batch 
        #  -> input_ids : (batch_size, input_len)
        #  -> attention_mask: (batch_size, input_len)
        #  -> labels: (batch_size x n_choices, max_choice_len)
        #  -> answer_index: (batch_size, )
        return {k: v.cuda() for k, v in new_batch.items()}
    
    def _broadcast_tensors(self, attention_mask, encoder_outputs, num_choices: int):
        attention_mask = torch.repeat_interleave(attention_mask, num_choices, dim=0)
        encoder_outputs.last_hidden_state = torch.repeat_interleave(encoder_outputs.last_hidden_state, num_choices, dim=0)
        return attention_mask, encoder_outputs
    
    def _compute_nll_of_all_choices(
            self,
            model: T5ForConditionalGeneration,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            labels: torch.LongTensor,
    ):
        """
            input_ids: (batch_size, input_seq_len)
            attention_mask: (batch_size, input_seq_len)
            labels: (batch_size x num_choices, max_choice_len)
        """
        batch_size = input_ids.shape[0]
        num_choices, max_choice_len = labels.shape[0] // batch_size, labels.shape[1]
        encoder_outputs = model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask, encoder_outputs = self._broadcast_tensors(attention_mask, encoder_outputs, num_choices)

        model_outputs = model( attention_mask=attention_mask, encoder_outputs=encoder_outputs, labels=labels,)
        logits = model_outputs.logits

        nll = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), reduction='none') # (batch_size x num_choices x max_choice_len)

        return nll

class MetricEvaluator(EvaluatorBase):

    def __init__(self, split: str, eval_batch_size: int):
        super(MetricEvaluator, self).__init__(split, eval_batch_size)

    def _compute_perplexity_of_all_choices(
            self,
            model: T5ForConditionalGeneration,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            labels: torch.LongTensor,
    ):
        """
            input_ids: (batch_size, input_seq_len)
            attention_mask: (batch_size, input_seq_len)
            labels: (batch_size x num_choices, max_choice_len)
        """
        nll = self._compute_nll_of_all_choices(model, input_ids, attention_mask, labels) # (batch_size x num_choices x max_choice_len)

        batch_size = input_ids.shape[0]
        num_choices, max_choice_len = labels.shape[0] // batch_size, labels.shape[1]
        label_mask = (labels != -100) # (batch_size x num_choices, max_choice_len)
        log_ppl = nll.reshape(batch_size * num_choices, max_choice_len).sum(dim=-1) / label_mask.sum(dim=-1) # (batch_size x num_choices)
        return log_ppl.reshape(batch_size, num_choices).exp() # (batch_size, num_choices)
    
    @torch.inference_mode()
    def evaluate(
            self,
            model: T5ForConditionalGeneration,
            tokenizer: AutoTokenizer,
            dataset: Dataset,
    ):
        metrics = dict()
        model.eval()
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        dataloader = DataLoader(dataset[self.split], batch_size=self.eval_batch_Size, shuffle=False, collate_fn=lambda x: self._nested_collate_fn(x, data_collator))
        total_n, accurate_n = 0, 0
        for batch in dataloader:
            ppl = self._compute_perplexity_of_all_choices(model, batch['input_ids'], batch['attention_mask'], batch['labels'])
            accurate_n += (batch['answer_index'] == torch.argmin(ppl, dim=-1)).sum().item()
            total_n += batch['input_ids'].shape[0]
        
        accuracy = accurate_n / total_n
        metrics.update({'accuracy': accuracy})

        # logger.info(f"eval metrics: {metrics}")

        return metrics
    
    def save_metrics(
            self,
            metrics: dict,
            path_to_save: str,
    ):
        # if path_to_save exists, update its value with metrics and resave.
        if os.path.exists(path_to_save):
            with open(path_to_save, "r") as f:
                _metrics = dict(json.load(f))
            _metrics.update(metrics)
            metrics = _metrics
        
        with open(path_to_save, "w") as f:
            json.dump(metrics, f)
        
        logger.info("saved metrics")
        return
    
class FisherEvaluator(EvaluatorBase):
    def __init__(self, num_samples: int, approximate_by_sampling: bool=False):
        super(FisherEvaluator, self).__init__(split='validation', eval_batch_size=1)
        self.eps = 1e-12
        self.num_samples = num_samples
        self.indicies_dict = dict()
        self.approximate_by_sampling = approximate_by_sampling
    
    def _compute_persample_diagonal_fisher(
            self,
            model: T5ForConditionalGeneration,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            labels: torch.LongTensor,
    ) -> Vector:
        """
            input_ids: (batch_size, input_seq_len)
            attention_mask: (batch_size, input_seq_len)
            labels: (batch_size x num_choices, max_choice_len)
        """
        assert input_ids.shape[0] == 1, "batch size must be 1 for computing per-sample diagonal fisher."
        nll = self._compute_nll_of_all_choices(model, input_ids, attention_mask, labels) # (batch_size x num_choices x max_choice_len)
        logprob = -nll.detach()
        batch_size = input_ids.shape[0]
        num_choices, max_choice_len = labels.shape[0] // batch_size, labels.shape[1]
        label_mask = (labels != -100)
        allchoice_logprob = logprob.view(num_choices, max_choice_len).sum(dim=-1) / label_mask.sum(dim=1)  # (num_choices)
        allchoice_prob = F.softmax(allchoice_logprob, dim=-1) # (num_choices)

        persample_fisher_vector = None
        if not self.approximate_by_sampling:
            """ compute diagonal fisher exactly """
            perchoice_labels = torch.split(labels, split_size_or_sections=1, dim=0) # tuple( 1 x 1, max_choice_len )
        else:
            """ compute diagonal fisher by sampling """
            choice_idx = np.random.choice(num_choices, size=1, p=allchoice_prob.cpu().numpy())
            perchoice_labels = tuple(labels[[choice_idx], :], )

        for i in range(len(perchoice_labels)):
            loss = -model(input_ids=input_ids, attention_mask=attention_mask, labels=perchoice_labels[i]).loss
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                current_fisher_vector = Vector(vector={pname: torch.square(pvalue.grad) for pname, pvalue in model.named_parameters()})
            persample_fisher_vector += allchoice_prob[i] * current_fisher_vector

        model.zero_grad()
        return persample_fisher_vector
        
    def compute_diagonal_fisher(
            self,
            model: T5ForConditionalGeneration,
            tokenizer: AutoTokenizer,
            dataset: Dataset,
            dataset_name: str,
            use_best_indicies: bool=False,
    ):
        model.eval()
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        if dataset_name not in self.indicies_dict:
            self.indicies_dict[dataset_name] = np.random.choice(len(dataset[self.split]), size=min(self.num_samples, len(dataset[self.split])), replace=False)
        indicies = self.indicies_dict[dataset_name]
        sampled_dataset = dataset[self.split].select(indicies)
        dataloader = DataLoader(sampled_dataset, batch_size=self.eval_batch_Size, shuffle=False, collate_fn=lambda x: self._nested_collate_fn(x, data_collator))

        fisher_vector = None
        for batch in dataloader:
            persample_fisher_vector = self._compute_persample_diagonal_fisher(model, batch['input_ids'], batch['attention_mask'], batch['labels'])
            fisher_vector += persample_fisher_vector
        
        fisher_vector /= self.num_samples
        fisher_vector += self.eps
        return fisher_vector