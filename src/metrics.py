import evaluate
import numpy as np

# Load the metric
metric = evaluate.load("seqeval")  # For sequence labeling tasks

# Define a function to compute metrics
def compute_metrics(p, entity_labels):
    predictions, labels = p
    # Convert the predicted label IDs to actual labels
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (usually -100 in the labels)
    true_labels = [[entity_labels[l] for l in label if l != -100] for label in labels]
    predicted_labels = [[entity_labels[pred] for pred, lab in zip(prediction, label) if lab != -100]
                        for prediction, label in zip(predictions, labels)]

    # Compute the metrics using seqeval
    results = metric.compute(predictions=predicted_labels, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
