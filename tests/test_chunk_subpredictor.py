import unittest

import numpy as np
import torch
from transformers import pipeline

from divide_and_conquer_sentiment.subprediction.sentence import Chunker, ChunkSubpredictor


class ChunkSubpredictorTest(unittest.TestCase):
    def setUp(self):
        """
        Set up a Divide instance for testing.
        """
        chunker = Chunker()  # Chunker returning an empty list
        sentiment_model = pipeline(
            "text-classification",
            model="j-hartmann/sentiment-roberta-large-english-3-classes",
            return_all_scores=True,
            device="cpu",
        )
        self.subpredictor = ChunkSubpredictor(chunker, sentiment_model)

    def test_predict_neutral_sentence(self):
        # Arrange
        res = self.subpredictor.predict(["I am neutral to it."])
        expected = torch.tensor(1)
        self.assertEqual(np.argmax(res[0]), expected)
