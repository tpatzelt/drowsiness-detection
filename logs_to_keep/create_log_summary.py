import csv
import json
from pathlib import Path

from drowsiness_detection.data import load_experiment_objects

with open('./log_summary.csv', 'w', newline='') as csvfile:
    fieldnames = ['id', 'model_name', 'window_in_sec', 'description', 'seed', 'test_size',
                  'model_parameter', 'train_acc', 'test_acc', 'cv_train_acc', 'cv_test_acc']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    e_id = 1
    log_dir = '../logs/'
    for log_dir_single in sorted(Path(log_dir).iterdir(),
                                 key=lambda x: int(x.name) if len(str(x)) < 4 else 0):
        if not log_dir_single.is_dir() or log_dir_single.name == '_sources':
            print("skipping: ", log_dir_single)
            continue
        e_id = log_dir_single.name
        try:
            config, best_model, search_results = load_experiment_objects(experiment_id=e_id, log_dir=log_dir)
        except FileNotFoundError:
            continue
        model_name = config['model_name']
        window_in_sec = config['window_in_sec']
        seed = config['seed']
        with open(f"{log_dir}{e_id}/info.json") as fp:
            info = json.load(fp)
        best_params = info['best_params']
        train_acc = info['train_accuracy']
        test_acc = info['test_accuracy']
        cv_train_acc = info['best_cv_train_accuracy']
        cv_test_acc = info['best_cv_test_accuracy']
        test_size = config['test_size']

        row = dict(id=e_id, model_name=model_name, window_in_sec=window_in_sec, seed=seed, model_parameter=best_params, train_acc=train_acc, test_acc=test_acc, cv_train_acc=cv_train_acc, cv_test_acc=cv_test_acc, description='', test_size=test_size)
        writer.writerow(rowdict=row)
