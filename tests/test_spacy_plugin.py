import pytest
import spacy
from benepar.integrations.spacy_plugin import BeneparComponent, NonConstituentException

MODEL_PATH = "/home/netuser/nltk_data/models/benepar_en3"

def test_spacy_integration():
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("benepar", config={"model": MODEL_PATH})
    
    doc = nlp("The time for action is now.")
    sent = list(doc.sents)[0]
    
    # Assert sentence extensions
    assert sent._.parse_string.startswith("(S")
    assert sent._.labels == ("S",)
    
    # Assert children and parents
    children = list(sent._.children)
    assert len(children) > 0
    assert children[0].text == "The time for action"
    
    assert sent._.parent is None
    
    # Assert token level extensions
    token = sent[0]
    assert token._.parent is not None
    assert token._.parse_string == "(DT The)"
    
    # Assert exception for non-constituent span
    non_constituent = sent[1:3]  # "time for"
    with pytest.raises(NonConstituentException):
        _ = non_constituent._.labels
