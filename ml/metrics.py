import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

def classification_metrics(labels, probabilities):
    predictions = [int(p >= .5) for p in probabilities]
    result = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }
    try: result["roc_auc"] = roc_auc_score(labels, probabilities)
    except ValueError: result["roc_auc"] = None
    return result

def save_confusion_matrix(labels, probabilities, output_path):
    matrix = confusion_matrix(labels, [int(p >= .5) for p in probabilities])
    figure, axis = plt.subplots()
    axis.imshow(matrix, cmap="Blues")
    axis.set_xlabel("Predicted"); axis.set_ylabel("Actual")
    figure.savefig(output_path, bbox_inches="tight"); plt.close(figure)

