import os
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import sys 
sys.path.append('.')
import config 


def main():
    # Load dataset info
    tfds_name = config.TFDS_NAME
    tdfs_info = tfds.builder(tfds_name).info

    test_split = 'test'
    if tfds_name == "glue/mnli":
        test_split = 'test_matched'

    # Load model 
    bert_type = config.BERT_MODEL_NAME.split('/')[-1]
    saved_model_name = f'{tfds_name.replace("/", "_")}_{bert_type}'
    saved_model_path = os.path.join(config.MODEL_SAVE_DIR, saved_model_name)

    print(f"Loading model from {saved_model_path}")
    model = tf.saved_model.load(saved_model_path)

    # Load test dataset
    in_memory_ds = tfds.load(tfds_name, batch_size=-1, shuffle_files=False)

    # Extract features
    sentence_features = list(tdfs_info.features.keys())
    sentence_features.remove('idx')
    if 'label' in sentence_features:
        sentence_features.remove('label')
    
    # For some GLUE tasks, test set doesn't have labels
    has_labels = 'label' in in_memory_ds[test_split]

    test_dataset = tf.data.Dataset.from_tensor_slices(in_memory_ds[test_split])

    correct = 0
    total = 0

    for i, test_example in enumerate(test_dataset.take(100)):
        # Prepare inputs based on mumber of features 
        if len(sentence_features) == 1:
            inputs = tf.constant([test_example[sentence_features[0]].numpy().decode()])
            result = model(inputs)
        else:
            inputs = [
                tf.constant([test_example[sentence_features[0]].numpy().decode()]),
                tf.constant([test_example[sentence_features[1]].numpy().decode()])
            ]
            result = model(inputs)
        
        predicted_class = tf.argmax(result, axis=1).numpy()[0]

        # If test set has labels, compute accuracy
        if has_labels:
            true_label = test_example['label'].numpy()
            if predicted_class == true_label:
                correct += 1
            total += 1
        
            if i < 5:
                print(f"Example {i+1}")
                for feature in sentence_features:
                    print(f" {feature}: {test_example[feature].numpy().decode()}")
                print(f" True label: {true_label}, Predicted: {predicted_class}")
                print(f" Raw prediction: {result.numpy()[0]}")
                print()
    
    # Report results
    if has_labels and total > 0:
        print(f"Evaluation accuracy on {total} examples: {correct/total:.4f}")
    else:
        print("Test set doesn't have lables, can't compute accuracy")
    
    # Save evaluation results
    results_path = os.path.join(config.MODEL_SAVE_PATH, f"{saved_model_name}_eval.txt")
    with open(results.path, "w") as f:
        if has_labels and total > 0:
            f.write(f"Accuracy: {correct/total:.4f}\n")
        f.write(f"Evaluated on {total} examples\n")


if __name__ == "__main__":
    main()