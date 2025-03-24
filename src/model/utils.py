import os
from transformers import T5ForConditionalGeneration, AutoTokenizer

def load_checkpoints(
        pt_model_root: str,
        ft_model_root: str,
        model_name_or_path: str,
        seed: int,
        datasets: list[str],
) -> tuple[T5ForConditionalGeneration, dict[str, T5ForConditionalGeneration], AutoTokenizer]:
    """
        load pre-trained model, dataset-fine-tuned model mapping, and tokenizer.
    """
    pt_model_path = os.path.join(pt_model_root, model_name_or_path)
    pt_model = T5ForConditionalGeneration.from_pretrained(pt_model_path).cuda()
    # assume identical tokenizer across pre-trained model and fine-tuned models.
    tokenizer = AutoTokenizer.from_pretrained(pt_model_path)

    ft_model_dict = dict()
    for dataset in datasets:
        ft_model_path = os.path.join(ft_model_root, model_name_or_path, dataset, f"seed{seed}")
        ft_model = T5ForConditionalGeneration.from_pretrained(ft_model_path).cuda()
        ft_model_dict[dataset] = ft_model
    
    return pt_model, ft_model_dict, tokenizer