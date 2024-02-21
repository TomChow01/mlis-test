import pytest
import os

from interview import CustomSLM

import torch


def get_data_dir():
    return os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "data"))


class TestingData:
    def __init__(self, path=os.path.join(get_data_dir(), "1000.txt")):
        self.path = path
        self.model = CustomSLM(path)

    def __eq__(self, other):
        return self.path == other.path
    


@pytest.fixture
def default_model():
    return TestingData()


def test_vocabulary_mask(default_model: TestingData):
    model = default_model.model
    
    tokens = model.tokenise("this is a test sentence, and this should work?")
    
    assert "input_ids" in tokens
    assert "attention_mask" in tokens
    
    for t in tokens["input_ids"].reshape(-1):
        assert model.vocabulary_mask[t]
    
    
def test_next_token_probabilities(default_model: TestingData):
    model = default_model.model
    
    tokenisation = model.tokenise("this cat is")
    
    probs = model.next_token_probabilities(tokenisation)
    
    assert pytest.approx(1.0) == torch.sum(probs).item()
    
def test_top_k_words(default_model: TestingData):
    model = default_model.model
    n = 4
    assert len(model.top_k_words("this cat is", n)) == 4
    
    
def test_top_n_sentences(default_model: TestingData):
    model = default_model.model
    
    root_sentence = "this cat is"
    n = 2
    k = 10 # don't use high k if using more tokens
    n_tokens = 2
    sentences = model.top_n_sentences(root_sentence, n, n_tokens, k)
    assert len(sentences) == n
    for s in sentences:
        assert s.startswith(root_sentence)
        
def test_text_probability(default_model: TestingData):
    model = default_model.model
    
    sentence = "the cat is black."
    log_prob = model.text_log_probability(sentence)
    assert log_prob < 0.0
    assert -torch.inf < log_prob
    

def test_sentence_probabilities(default_model: TestingData):
    
    model = default_model.model
    
    colours = ["black", "white", "blue", "red", "gray", "purple"]
    sentences = [f"the cat is {c}." for c in colours]
    
    log_probs = [model.text_log_probability(s) for s in sentences]
    
    assert abs(log_probs[0] - (-17)) < 0.5
    assert abs(log_probs[-2] - (-20)) < 0.5
    assert torch.isneginf(torch.Tensor([log_probs[-1]])).item() # purple is not in the word list
    
    assert torch.all(torch.Tensor(sorted(log_probs, reverse=True)) == torch.Tensor(log_probs)).item()
    