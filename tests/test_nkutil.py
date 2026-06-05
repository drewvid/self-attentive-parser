import argparse
import pytest
from benepar.nkutil import HParams

def test_hparams_init():
    h = HParams(a=1, b="test", c=True)
    assert h.a == 1
    assert h.b == "test"
    assert h.c is True

def test_hparams_get_set_item():
    h = HParams(a=1)
    assert h["a"] == 1
    h["a"] = 2
    assert h["a"] == 2
    
    with pytest.raises(KeyError):
        h["nonexistent"] = 3

def test_hparams_to_dict():
    h = HParams(a=1, b="test", _internal=10)
    d = h.to_dict()
    assert d == {"a": 1, "b": "test"}
    assert "_internal" not in d

def test_hparams_populate_and_set():
    h = HParams(learning_rate=0.01, use_encoder=True, force_root=False)
    parser = argparse.ArgumentParser()
    h.populate_arguments(parser)
    
    # Check default parsing
    args = parser.parse_args([])
    assert args.learning_rate == 0.01
    assert getattr(args, "use_encoder", None) is None  # wait, populate_arguments uses --no-use-encoder if default is True?
    # Let's check how bool is handled in populate_arguments:
    # if isinstance(v, bool):
    #     if not v:
    #         parser.add_argument(f"--{k}", action="store_true")
    #     else:
    #         parser.add_argument(f"--no-{k}", action="store_false")
    # Yes! Since use_encoder is True (which is v), it adds --no-use-encoder with action="store_false".
    # And force_root is False (not v), it adds --force-root with action="store_true".
    # Under argparse, store_false defaults to True, store_true defaults to False.
    assert args.no_use_encoder is True
    assert args.force_root is False
    
    args_custom = parser.parse_args(["--no-use-encoder", "--force-root"])
    assert args_custom.no_use_encoder is False
    assert args_custom.force_root is True
    
    h.set_from_args(args_custom)
    assert h.use_encoder is False
    assert h.force_root is True


def test_hparams_private_keys():
    import argparse
    h = HParams(a=1, _internal=10)
    parser = argparse.ArgumentParser()
    h.populate_arguments(parser)
    
    args = parser.parse_args([])
    assert hasattr(args, "a")
    assert not hasattr(args, "no_internal")
    assert not hasattr(args, "internal")
    assert not hasattr(args, "_internal")

