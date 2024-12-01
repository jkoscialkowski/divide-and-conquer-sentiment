import unittest
from src.divide import Divide  # Assuming your Divide class is saved in `divide.py`

class TestDivide(unittest.TestCase):
    def setUp(self):
        """
        Set up a Divide instance for testing.
        """
        self.divide = Divide()

    def test_identity(self):
        """
        Test the segmentation of text into sentences.
        """
        text = "This is the only sentence."
        expected = ["This is the only sentence."]
        result = self.divide.segment_text(text)
        self.assertEqual(result, expected)

    def test_segment_text(self):
        """
        If sentence is not the last in text, additional whitespace is added"
        """
        text = "This is the first sentence. Here's the second one!"
        expected = ["This is the first sentence. ", "Here's the second one!"]
        result = self.divide.segment_text(text)
        self.assertEqual(result, expected)

    def test_3_setnceces(self):
        """
        Test the segmentation of text into 3 sentences.
        """
        text = "This is the first sentence. Here's the second one! And here's the third one!"
        expected = ["This is the first sentence. ", "Here's the second one! ", "And here's the third one!"]
        result = self.divide.segment_text(text)
        self.assertEqual(result, expected)

    def test_extract_clauses_simple(self):
        """
        Test clause extraction with simple sentences (no clauses).
        """
        text = "This is a simple sentence."
        expected = ["This is a simple sentence"]
        result = self.divide.extract_clauses(text)
        self.assertEqual(result, expected)

    def test_extract_clauses_with_clauses(self):
        """
        This test shows that clausie is not working very well
        """
        text = "I think she is coming, but I am not sure."
        # Expected clauses depend on how `claucy` handles the sentence
        # Adjust this if the output differs for your setup
        expected = ["I thought coming", "she is coming", "I am sure"]
        result = self.divide.extract_clauses(text)
        self.assertEqual(result, expected)

    def test_extract_clauses_empty_text(self):
        """
        Test clause extraction with empty text.
        """
        text = ""
        expected = []
        result = self.divide.extract_clauses(text)
        self.assertEqual(result, expected)

    def test_extract_clauses_error_handling(self):
        """
        Test that the function handles in clause extraction gracefully.
        """
        text = "This is a test sentence with invalid clauses."
        # Mock or simulate an AttributeError if needed in the `divide.py` code
        # Here we assume it proceeds without crashing and returns the sentence
        expected = ["This is a test sentence"]
        result = self.divide.extract_clauses(text)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()

