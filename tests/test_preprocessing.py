import tensorflow as tf
import pytest
import sys
sys.path.append('.')
from src.preprocessing import make_bert_preprocess_model

def test_preprocessing_model_creation():
    # Test that the preprocessing model can be created
    preprocess_model = make_bert_preprocess_model(['text1', 'text2'])
    assert isinstance(preprocess_model, tf.keras.Model)
    
    # Test that the model has the expected inputs
    assert len(preprocess_model.inputs) == 2
    assert preprocess_model.inputs[0].name == 'text1:0'
    assert preprocess_model.inputs[1].name == 'text2:0'
    
    # Test that the model produces the expected outputs
    test_inputs = [tf.constant(['Hello']), tf.constant(['World'])]
    outputs = preprocess_model(test_inputs)
    assert 'input_word_ids' in outputs
    assert 'input_mask' in outputs
    assert 'input_type_ids' in outputs