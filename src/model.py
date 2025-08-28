import tensorflow as tf
import tensorflow_hub as hub

def get_encoder_handle(bert_model_name):
    """Get the appropriate encoder model handle for a BERT model."""
    map_name_to_handle = {
        'bert_en_uncased_L-12_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
        # Include other mappings from the notebook
    }
    return map_name_to_handle[bert_model_name]

def build_classifier_model(bert_model_name, num_classes):
    """Build a classifier model with BERT encoder."""
    tfhub_handle_encoder = get_encoder_handle(bert_model_name)
    
    class Classifier(tf.keras.Model):
        def __init__(self, num_classes):
            super(Classifier, self).__init__(name="prediction")
            self.encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True)
            self.dropout = tf.keras.layers.Dropout(0.1)
            self.dense = tf.keras.layers.Dense(num_classes)

        def call(self, preprocessed_text):
            encoder_outputs = self.encoder(preprocessed_text)
            pooled_output = encoder_outputs["pooled_output"]
            x = self.dropout(pooled_output)
            x = self.dense(x)
            return x

    model = Classifier(num_classes)
    return model