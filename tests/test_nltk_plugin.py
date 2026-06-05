import pytest
import nltk
from benepar import Parser, InputSentence

MODEL_PATH = "/home/netuser/nltk_data/models/benepar_en3"

def test_input_sentence():
    # Test simple word-based initialization
    sent = InputSentence(words=["Fly", "safely", "."])
    assert sent.words == ["Fly", "safely", "."]
    assert sent.tree is None
    
    # After filling missing fields, pos() uses escaped_words
    sent_with_leaves = InputSentence(escaped_words=["Fly", "safely", "."])
    assert sent_with_leaves.pos() == [("Fly", "UNK"), ("safely", "UNK"), (".", "UNK")]

def test_nltk_parser_basics():
    parser = Parser(MODEL_PATH)
    
    # Test parsing List[str]
    tree = parser.parse(["The", "time", "for", "action", "is", "now", "."])
    assert isinstance(tree, nltk.Tree)
    assert tree.label() == "TOP"
    
    # Test parsing InputSentence
    sent = InputSentence(words=["The", "time", "for", "action", "is", "now", "."])
    tree2 = parser.parse(sent)
    assert isinstance(tree2, nltk.Tree)
    assert tree2.label() == "TOP"
    
    # Test parsing raw string
    tree3 = parser.parse("The time for action is now.")
    assert isinstance(tree3, nltk.Tree)
    assert tree3.label() == "TOP"

def test_nltk_parser_batch():
    parser = Parser(MODEL_PATH)
    sents = [
        ["The", "time", "for", "action", "is", "now", "."],
        ["It", "is", "never", "too", "late", "."]
    ]
    trees = list(parser.parse_sents(sents))
    assert len(trees) == 2
    assert all(isinstance(t, nltk.Tree) for t in trees)
