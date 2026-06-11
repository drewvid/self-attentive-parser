import os
import pytest
import torch
from benepar.parse_chart import ChartParser

MODEL_PATH = os.path.expanduser("~/nltk_data/models/benepar_en3")

class DummyExample:
    def __init__(self, words, space_after):
        self.words = words
        self.space_after = space_after
        self.tree = None

    def pos(self):
        return [(w, "UNK") for w in self.words]

def test_chart_parser_loading():
    parser = ChartParser.from_trained(MODEL_PATH)
    assert isinstance(parser, ChartParser)
    
    # Check vocabularies and configurations
    assert parser.tag_vocab is not None
    assert parser.label_vocab is not None
    assert parser.d_model > 0
    assert isinstance(parser.device, torch.device)

def test_chart_parser_parse():
    parser = ChartParser.from_trained(MODEL_PATH)
    
    words = ["The", "time", "for", "action", "is", "now", "."]
    space_after = [True, True, True, True, True, False, False]
    example = DummyExample(words, space_after)
    
    # Run parse
    trees = parser.parse([example])
    assert len(trees) == 1
    
    tree = trees[0]
    assert tree.label() == "TOP"
    assert tree.leaves() == words
