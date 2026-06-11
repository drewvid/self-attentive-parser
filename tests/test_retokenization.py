import os
import pytest
import torch
import transformers
from benepar.retokenization import retokenize, Retokenizer

MODEL_PATH = os.path.expanduser("~/nltk_data/models/benepar_en3")

def test_retokenize_func():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, fast=True)
    words = ["The", "time", "for", "action", "is", "now", "."]
    space_after = [True, True, True, True, True, False, False]
    
    result = retokenize(tokenizer, words, space_after)
    assert "input_ids" in result
    assert "words_from_tokens" in result
    assert len(result["words_from_tokens"]) == len(words)

def test_retokenizer_class():
    retokenizer = Retokenizer(MODEL_PATH, retain_start_stop=True)
    
    assert retokenizer.is_t5 is True
    
    words = ["The", "time", "for", "action", "is", "now", "."]
    space_after = [True, True, True, True, True, False, False]
    
    encoded = retokenizer(words, space_after)
    assert "input_ids" in encoded
    assert "words_from_tokens" in encoded
    
    # Test pad functionality
    batch = [encoded]
    padded = retokenizer.pad(batch, return_tensors="pt")
    assert "input_ids" in padded
    assert "words_from_tokens" in padded
    assert "valid_token_mask" in padded


def test_retokenize_with_whitespace():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, fast=True)
    
    # Text with leading/trailing/multiple whitespaces and newlines
    # s = "\n The   time  \n"
    words = ["\n", "The", " ", "time", "\n"]
    space_after = [True, True, True, True, False]
    
    result = retokenize(tokenizer, words, space_after)
    assert len(result["words_from_tokens"]) == len(words)
    # The whitespace words should be unmapped (-100)
    assert result["words_from_tokens"][0] == -100
    assert result["words_from_tokens"][2] == -100
    assert result["words_from_tokens"][4] == -100
    # Actual words should be mapped to valid subword token indices >= 0
    assert result["words_from_tokens"][1] >= 0
    assert result["words_from_tokens"][3] >= 0

