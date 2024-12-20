import claucy
import pysbd
import spacy
import torch
from transformers import TextClassificationPipeline

from .base import SubpredictorBase


class ChunkSubpredictor(SubpredictorBase):
    def __init__(self, chunker: "Chunker", sentiment_model: TextClassificationPipeline):
        if sentiment_model._postprocess_params["top_k"] is not None:
            raise ValueError("The sentiment model must return all scores.")

        self.chunker = chunker
        self.sentiment_model = sentiment_model

    def predict(self, inputs: list[str]) -> list[torch.Tensor]:
        chunked_sentences = self.chunker.chunk_list(inputs)
        return list(map(self._chunk_to_tensor, chunked_sentences))

    def _chunk_to_tensor(self, chunk: list[str]) -> torch.Tensor:
        return torch.tensor([[pred["score"] for pred in chunk_preds] for chunk_preds in self.sentiment_model(chunk)])


class Chunker:
    def __init__(self, language="en", spacy_model="en_core_web_sm"):
        """
        Initialize the Chunker class with a segmenter and a SpaCy pipeline enhanced with claucy.

        :param language: Language for the segmenter.
        :param spacy_model: SpaCy language model to load.
        """
        self.segmenter = pysbd.Segmenter(language=language, clean=False)
        self.nlp = spacy.load(spacy_model)
        claucy.add_to_pipe(self.nlp)

    def chunk_text(self, text):
        """
        Chunk text into sentences using pysbd.

        :param text: Input text.
        :return: List of sentences.
        """
        return self.segmenter.segment(text)

    def chunk_list(self, texts):
        """
        Chunk a list of texts into sentences using pysbd.

        :param texts: List of input texts.
        :return: List of lists, where each sublist contains the sentences of the corresponding input text.
        """
        segmented_texts = [self.chunk_text(text) for text in texts]
        return segmented_texts

    def extract_clauses(self, text):
        """
        Extract clauses from the text using claucy.

        :param text: Input text.
        :return: List of extracted clauses or original sentences if no clauses found.
        """
        sentences = self.chunk_text(text)
        clauses = []

        for sentence in sentences:
            doc = self.nlp(sentence)
            if len(doc._.clauses) == 0:
                # If no clauses, append the sentence itself
                clauses.append(doc.text)
            else:
                subsentences = []
                for sub in doc._.clauses:
                    try:
                        subsentences.append(sub.to_propositions(as_text=True)[0])
                    except AttributeError:
                        print("AttributeError during conversion to propositions:", sub)
                clauses.extend(subsentences)
        return clauses
