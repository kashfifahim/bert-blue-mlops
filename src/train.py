import os
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from official.nlp import optimization
import sys
sys.path.append('.')
from src.preprocessing import make_bert_preprocess_model
from src.model import build_classifier_model
import config

def get_configuration(glue_task):
    """Get the appropriate metrics and loss for a GLUE task."""
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if glue_task == 'glue/cola':
        metrics = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)
    else:
        metrics = tf.keras.metrics.SparseCategoricalAccuracy(
            'accuracy', dtype=tf.float32)

    return metrics, loss

def load_dataset_from_tfds(in_memory_ds, info, split, batch_size, bert_preprocess_model):
    """Load and preprocess a dataset from TFDS."""
    is_training = split.startswith('train')
    dataset = tf.data.Dataset.from_tensor_slices(in_memory_ds[split])
    num_examples = info.splits[split].num_examples

    if is_training:
        dataset = dataset.shuffle(num_examples)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda ex: (bert_preprocess_model(ex), ex['label']))
    dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset, num_examples

def main():
    # Load dataset info
    tfds_info = tfds.builder(config.TFDS_NAME).info
    
    # Extract features and splits
    sentence_features = list(tfds_info.features.keys())
    sentence_features.remove('idx')
    sentence_features.remove('label')
    
    train_split = 'train'
    validation_split = 'validation'
    if config.TFDS_NAME == 'glue/mnli':
        validation_split = 'validation_matched'
    
    num_classes = tfds_info.features['label'].num_classes
    
    # Load dataset into memory
    in_memory_ds = tfds.load(config.TFDS_NAME, batch_size=-1, shuffle_files=True)
    
    # Create preprocessing model
    bert_preprocess_model = make_bert_preprocess_model(sentence_features, config.SEQ_LENGTH)
    
    # Set up distribution strategy
    strategy = tf.distribute.MirroredStrategy() if tf.config.list_physical_devices('GPU') else tf.distribute.get_strategy()
    
    with strategy.scope():
        # Get metrics and loss
        metrics, loss = get_configuration(config.TFDS_NAME)
        
        # Prepare datasets
        train_dataset, train_data_size = load_dataset_from_tfds(
            in_memory_ds, tfds_info, train_split, config.BATCH_SIZE, bert_preprocess_model)
        steps_per_epoch = train_data_size // config.BATCH_SIZE
        num_train_steps = steps_per_epoch * config.EPOCHS
        num_warmup_steps = num_train_steps // 10
        
        validation_dataset, validation_data_size = load_dataset_from_tfds(
            in_memory_ds, tfds_info, validation_split, config.BATCH_SIZE, bert_preprocess_model)
        validation_steps = validation_data_size // config.BATCH_SIZE
        
        # Build and compile model
        classifier_model = build_classifier_model(config.BERT_MODEL_NAME, num_classes)
        
        optimizer = optimization.create_optimizer(
            init_lr=config.INIT_LR,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            optimizer_type='adamw')
        
        classifier_model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
        
        # Train model
        history = classifier_model.fit(
            x=train_dataset,
            validation_data=validation_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=config.EPOCHS,
            validation_steps=validation_steps)
        
        # Save model for inference
        bert_type = config.BERT_MODEL_NAME.split('/')[-1]
        saved_model_name = f'{config.TFDS_NAME.replace("/", "_")}_{bert_type}'
        saved_model_path = os.path.join(config.MODEL_SAVE_PATH, saved_model_name)
        
        # Create exportable model with preprocessing
        preprocess_inputs = bert_preprocess_model.inputs
        bert_encoder_inputs = bert_preprocess_model(preprocess_inputs)
        bert_outputs = classifier_model(bert_encoder_inputs)
        model_for_export = tf.keras.Model(preprocess_inputs, bert_outputs)
        
        # Save model
        os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
        model_for_export.save(saved_model_path, include_optimizer=False)
        
        print(f"Model saved to {saved_model_path}")
        
        # Save training metrics for tracking
        metrics_path = os.path.join(config.MODEL_SAVE_PATH, f"{saved_model_name}_metrics.txt")
        with open(metrics_path, "w") as f:
            for key, value in history.history.items():
                f.write(f"{key}: {value[-1]}\n")

if __name__ == "__main__":
    main()