from abc import ABC, abstractmethod

import torch
from setfit.span.modeling import PolarityModel
from transformers import pipeline

class SentimentModel(ABC):
    """
       Abstract base class for sentiment analysis models.
    """

    @abstractmethod
    def predict(self, subsentences: list[str]) -> list[torch.Tensor]:
        """
        Predict the sentiment for each subsentence.

        :param subsentences: A list of strings, where each string is a subsentence to score.
        :return: A list of torch.Tensor objects of a size 1x3, where each tensor represents the sentiment
                 prediction for the corresponding subsentence.
        """
        pass

class PolaritySentimentModel(SentimentModel):

    def __init__(self, sentiment_model: PolarityModel):
        self.model = sentiment_model

    def predict(self, subsentences: list[str]) -> list[torch.Tensor]:
        """
            Polarity model returns a torch matrix of a size len(subsentences) x 4,
            where 4 represents probability of following classes:
            1. confilct, 2. negative, 3. neutral, 4. positive
            We decided to skip conflict class to make the downstream methods to work on one size input of n x 3.
            The conflict class occured rarly on examined datasets.
        """
        try:
            torch_matrix = self.model.predict_proba(subsentences, show_progress_bar=False)
        except ValueError:
            print(f'Prediction failed for subsentences {subsentences}, len = {len(subsentences)}')
            raise ValueError
        tensor_list = [torch_matrix[i][1:4].unsqueeze(0) for i in range(torch_matrix.size(0))]
        return tensor_list

class RobertaSentimentModel(SentimentModel):
    def __init__(self, sentiment_model: pipeline):
        self.model = sentiment_model
    def predict(self, subsentences: list[str]) -> list[torch.Tensor]:
        try:
            roberta_subpreds = self.model(subsentences)
        except ValueError:
            print(f'Prediction failed for subsentences {subsentences}, len = {len(subsentences)}')
            raise ValueError
        tensor_list = [torch.tensor([[x[0]['score'], x[1]['score'], x[2]['score']]]) for x in roberta_subpreds]
        return tensor_list