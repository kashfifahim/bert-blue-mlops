import os

# Model configuration
BERT_MODEL_NAME = os.environ.get('BERT_MODEL_NAME', 'bert_en_uncased_L-12_H-768_A-12')
TFDS_NAME = os.environ.get('TFDS_NAME', 'glue/cola')
EPOCHS = int(os.environ.get('EPOCHS', 3))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 32))
INIT_LR = float(os.environ.get('INIT_LR', 2e-5))
SEQ_LENGTH = int(os.environ.get('SEQ_LENGTH', 128))

# Paths
MODEL_SAVE_PATH = os.environ.get('MODEL_SAVE_PATH', './models')