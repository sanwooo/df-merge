import torch
from transformers import T5ForConditionalGeneration

class Vector:

    def __init__(self, pt_model: T5ForConditionalGeneration=None, ft_model: T5ForConditionalGeneration=None, vector: dict=None):

        if vector:
            self.vector = vector
        else:
            assert pt_model
            self.vector = dict()
            pt_pdict = self._get_pdict(pt_model)
            if ft_model:
                ft_pdict = self._get_pdict(ft_model)
                assert self._check_pname_match(pt_pdict, ft_pdict)
            for pname, pvalue in pt_pdict.items():
                if ft_model:
                    self.vector[pname] = ft_pdict[pname] - pt_pdict[pname]
                else:
                    self.vector[pname] = pt_pdict[pname]
    
    def _get_pdict(self, model: T5ForConditionalGeneration):
        return {pname: pvalue.data for pname, pvalue in model.named_parameters()}
    
    def _check_pname_match(self, pdict_a: dict[str, torch.nn.Parameter], pdict_b: dict[str, torch.nn.Parameter]):
        if set(pdict_a.keys()) == set(pdict_b.keys()):
            return True
        return False
        
    def apply_to(self, model: T5ForConditionalGeneration):
        """
            copy task vector parameter values to the model.
        """
        for pname, pvalue in model.named_parameters():
            pvalue.data.copy_(self.vector[pname])
        # model.load_state_dict(self.vector, strict=False)
        return model
    
    def __add__(self, other):
        # assert self._check_pname_match(self.vector, other.vector)
        new_vector = dict()
        if other is None:
            return self
        elif isinstance(other, int) or isinstance(other, float) or isinstance(other, torch.Tensor):
            for pname in self.vector:
                new_vector[pname] = self.vector[pname] + other
        else: # if other is task vector
            for pname in self.vector:
                new_vector[pname] = self.vector[pname] + other.vector[pname]
        return Vector(vector=new_vector)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        new_vector = dict()
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, torch.Tensor):
            for pname in self.vector:
                new_vector[pname] = other * self.vector[pname]
        else: # if other is task vector
            for pname in self.vector:
                new_vector[pname] = self.vector[pname] * other.vector[pname]
        return Vector(vector=new_vector)

    def __rmul__(self, other):
        return self.__mul__(other)


    def __truediv__(self, other):
        new_vector = dict()
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, torch.Tensor):
            for pname in self.vector:
                new_vector[pname] = self.vector[pname] / other
        else: # if other is task vector
            for pname in self.vector:
                new_vector[pname] = self.vector[pname] / other.vector[pname]
        return Vector(vector=new_vector)