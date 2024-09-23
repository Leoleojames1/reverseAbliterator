import unittest
from src.reverse_abliterator.reverseAbliterator import ReverseAbliterator

class TestReverseAbliterator(unittest.TestCase):
    def setUp(self):
        self.model_path = "path/to/test/model"
        self.target_instructions = ["Write a poem about nature", "Explain quantum physics"]
        self.baseline_instructions = ["Hello", "What's the weather like?"]

    def test_initialization(self):
        ra = ReverseAbliterator(
            model=self.model_path,
            dataset=([self.target_instructions, self.baseline_instructions]),
            device="cpu"
        )
        self.assertIsInstance(ra, ReverseAbliterator)

if __name__ == '__main__':
    unittest.main()
