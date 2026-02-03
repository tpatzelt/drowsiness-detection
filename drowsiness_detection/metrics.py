"""Classification metrics for evaluating model performance."""

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score


def calc_classification_metrics(y_trues: list, y_preds: list) -> tuple:
    """Calculate classification metrics for binary classification.
    
    Computes accuracy, recall, precision, and ROC AUC score for each pair of
    true and predicted labels. Values are binarized at threshold 0.5.
    
    Args:
        y_trues: List of true labels or probabilities
        y_preds: List of predicted labels or probabilities
        
    Returns:
        Tuple of (accuracies, recalls, precisions, aucs) where each is a list
    """
    y_trues = [y_true > 0.5 for y_true in y_trues]
    y_preds = [y_pred > 0.5 for y_pred in y_preds]
    accs = [accuracy_score(y_true, y_pred) for y_true, y_pred in zip(y_trues, y_preds)]
    recalls = [recall_score(y_true, y_pred) for y_true, y_pred in zip(y_trues, y_preds)]
    precisions = [precision_score(y_true, y_pred) for y_true, y_pred in zip(y_trues, y_preds)]
    aucs = [roc_auc_score(y_true, y_pred) for y_true, y_pred in zip(y_trues, y_preds)]
    return accs, recalls, precisions, aucs


def calc_mean_and_std_of_classification_metrics(y_trues: list, y_preds: list) -> tuple:
    """Calculate mean and standard deviation of classification metrics.
    
    Args:
        y_trues: List of true labels or probabilities
        y_preds: List of predicted labels or probabilities
        
    Returns:
        Tuple of tuples: ((mean_acc, std_acc), (mean_recall, std_recall), 
                          (mean_precision, std_precision), (mean_auc, std_auc))
    """
    accs, recalls, precisions, aucs = calc_classification_metrics(y_trues, y_preds)

    mean_acc = np.mean(accs)
    mean_recall = np.mean(recalls)
    mean_precision = np.mean(precisions)
    mean_auc = np.mean(aucs)

    std_acc = np.std(accs)
    std_recall = np.std(recalls)
    std_precision = np.std(precisions)
    std_auc = np.std(aucs)
    return (mean_acc, std_acc), (mean_recall, std_recall), (mean_precision, std_precision), (
        mean_auc, std_auc)


def print_metric_results(results: tuple) -> None:
    """Print formatted classification metric results.
    
    Args:
        results: Tuple of (mean, std) pairs for each metric from
                 calc_mean_and_std_of_classification_metrics()
    """
    results = [(round(x, 2), round(y, 3)) for x, y in results]
    (mean_acc, std_acc), (mean_recall, std_recall), (mean_precision, std_precision), (
        mean_auc, std_auc) = results
    print(rf"Mean Accuracy = {mean_acc} ± {std_acc}")
    print(rf"Mean Precision = {mean_precision} ± {std_precision}")
    print(rf"Mean Recall = {mean_recall} ± {std_recall}")
    print(rf"Mean ROC AUC = {mean_auc} ± {std_auc}")
