import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

def get_preprocessing_handle(bert_model_name):
    """Get the appropriate preprocessing model handle for a BERT model."""
    map_model_to_preprocess = {
        'bert_en_uncased_L-12_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        # Include other mappings from the notebook
    }
    return map_model_to_preprocess[bert_model_name]

def make_bert_preprocess_model(sentence_features, seq_length=128):
    """Returns Model mapping string features to BERT inputs."""
    input_segments = [
        tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)
        for ft in sentence_features
    ]

    # Tokenize the text to word pieces
    tfhub_handle_preprocess = get_preprocessing_handle(bert_model_name)
    bert_preprocess = hub.load(tfhub_handle_preprocess)
    tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')
    segments = [tokenizer(s) for s in input_segments]

    # Pack inputs
    packer = hub.KerasLayer(bert_preprocess.bert_pack_inputs,
                          arguments=dict(seq_length=seq_length),
                          name='packer')
    model_inputs = packer(segments)
    return tf.keras.Model(input_segments, model_inputs)