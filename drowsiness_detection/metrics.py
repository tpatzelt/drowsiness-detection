import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score


def calc_classification_metrics(y_trues, y_preds):
    y_trues = [y_true > .5 for y_true in y_trues]
    y_preds = [y_pred > .5 for y_pred in y_preds]
    accs = [accuracy_score(y_true, y_pred) for y_true, y_pred in zip(y_trues, y_preds)]
    recalls = [recall_score(y_true, y_pred) for y_true, y_pred in zip(y_trues, y_preds)]
    precisions = [precision_score(y_true, y_pred) for y_true, y_pred in zip(y_trues, y_preds)]
    aucs = [roc_auc_score(y_true, y_pred) for y_true, y_pred in zip(y_trues, y_preds)]
    return accs, recalls, precisions, aucs


def calc_mean_and_std_of_classification_metrics(y_trues, y_preds):
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


def print_metric_results(results):
    results = [(round(x, 2), round(y, 3)) for x, y in results]
    (mean_acc, std_acc), (mean_recall, std_recall), (mean_precision, std_precision), (
    mean_auc, std_auc) = results
    print(rf"Mean Accuracy = {mean_acc} ± {std_acc}")
    print(rf"Mean Precision = {mean_precision} ± {std_precision}")
    print(rf"Mean Recall = {mean_recall} ± {std_recall}")
    print(rf"Mean ROC AUC = {mean_auc} ± {std_auc}")
