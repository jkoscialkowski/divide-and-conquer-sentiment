import spacy
import claucy
import pysbd


class Divide:
    def __init__(self, language="en", spacy_model="en_core_web_sm"):
        """
        Initialize the Divide class with a segmenter and a SpaCy pipeline enhanced with claucy.

        :param language: Language for the segmenter.
        :param spacy_model: SpaCy language model to load.
        """
        self.segmenter = pysbd.Segmenter(language=language, clean=False)
        self.nlp = spacy.load(spacy_model)
        claucy.add_to_pipe(self.nlp)

    def segment_text(self, text):
        """
        Segment text into sentences using pysbd.

        :param text: Input text.
        :return: List of sentences.
        """
        return self.segmenter.segment(text)

    def extract_clauses(self, text):
        """
        Extract clauses from the text using claucy.

        :param text: Input text.
        :return: List of extracted clauses or original sentences if no clauses found.
        """
        sentences = self.segment_text(text)
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

