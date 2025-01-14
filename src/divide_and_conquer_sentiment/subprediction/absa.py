import torch
from setfit import AbsaModel

from .base import SubpredictorBase


class ABSASubpredictor(SubpredictorBase):
    def __init__(self, absa_model: AbsaModel):
        self.absa_model = absa_model

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(AbsaModel.from_pretrained(*args, **kwargs))

    def predict(self, inputs: list[str]) -> list[torch.Tensor]:
        docs, aspects_list = self.absa_model.aspect_extractor(inputs)
        if sum(aspects_list, []) == []:
            return aspects_list

        aspects_list = self.absa_model.aspect_model(docs, aspects_list)
        if sum(aspects_list, []) == []:
            return aspects_list

        inputs_list = list(self.absa_model.polarity_model.prepend_aspects(docs, aspects_list))
        preds = self.absa_model.polarity_model.predict_proba(inputs_list)
        iter_preds = iter(preds)

        preds_with_empty = [[next(iter_preds) for _ in aspects] for aspects in aspects_list]
        preds = []
        dtype = torch.get_default_dtype()
        for input, pred in zip(inputs, preds_with_empty):
            if len(pred) == 0:
                preds.append(self.absa_model.polarity_model.predict_proba(input).reshape(1, -1).to(dtype))
            else:
                preds.append(torch.vstack(pred).to(dtype))

        return preds
