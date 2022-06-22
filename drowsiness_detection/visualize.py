from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib import cm
from matplotlib.widgets import Slider
from scipy.interpolate import griddata
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV

from drowsiness_detection import config
from drowsiness_detection.data import get_session_idx, get_subject_idx
from drowsiness_detection.data import (load_experiment_objects, load_experiment_objects_nn)
from drowsiness_detection.data import (load_preprocessed_train_test_splits,
                                       session_type_mapping)


def plot_acc_and_loss(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.title('model accuracy/loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'loss'], loc='upper left')
    plt.show()


def generate_blink_animation(data: np.array, name: str, n_frames: int = 60):
    """Generates a html video plotting the eye closure signal
    over time. The range of the displayed x-axis can be controlled with n_frames."""
    fig = plt.figure(figsize=(12, 5))
    ax = plt.subplot(1, 1, 1)

    ax.set_xlim((-30, 30))
    ax.set_ylim((0, 1))
    ax.set_xlabel("Frames")
    ax.set_ylabel("Eye Closure")

    txt_title = ax.set_title("")
    line, = ax.plot([], [], 'r', lw=2)

    # ax.legend(["eye closure signal"])

    def drawframe(n):
        line.set_data(np.arange(n - n_frames // 2, n + n_frames // 2),
                      data[n - n_frames // 2:n + n_frames // 2])
        txt_title.set_text(f"Frame = {n}")
        ax.set_xlim((n - n_frames // 2, n + n_frames // 2))
        return line,

    anim = animation.FuncAnimation(fig, drawframe,
                                   frames=range(n_frames // 2, len(data) - n_frames // 2),
                                   interval=20, blit=True)
    writervideo = animation.FFMpegWriter(fps=30)
    return anim.save(f"{name}.mp4", writer=writervideo)


def show_frame_slider(data: np.array, n_frames: int = 60):
    """Shows a windows of n_frames from data. An interactive slider can
    be used to move through the data horizontally."""
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    ax.set_xlim((0, n_frames))
    ax.set_ylim((0, 1))
    ax.set_xlabel("Frames")
    ax.set_ylabel("Eye Closure")

    x = np.arange(0, n_frames)
    y = data[0:n_frames]
    line, = ax.plot(x, y, 'r', lw=2)
    vline = ax.axvline(x=n_frames // 2, linestyle="-")

    ax_frames = plt.axes([0.25, 0.1, 0.65, 0.03])

    allowed_frames = np.arange(n_frames // 2, len(data) - n_frames // 2,
                               step=2)  # does the video frame start at 0 or 1?

    sframes = Slider(
        ax_frames, "Frame", valmin=n_frames // 2, valmax=len(data) - n_frames // 2,
        valinit=n_frames // 2, valstep=allowed_frames)

    def update(val):
        n = val
        x = np.arange(n - n_frames // 2, n + n_frames // 2)
        y = data[n - n_frames // 2:n + n_frames // 2]
        line.set_data(x, y)
        ax.set_xlim((n - n_frames // 2, n + n_frames // 2))
        vline.set_xdata(n)
        fig.canvas.draw_idle()

    sframes.on_changed(update)
    return sframes, ax


def plot_roc_over_n_folds(classifier, X, y, cv=6, fit_model=False, ax=None):
    if isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    if ax is None:
        fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        if fit_model:
            classifier.fit(X[train], y[train])
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X[test],
            y[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"ROC for {classifier}",
    )
    ax.legend(loc="lower right")
    # plt.show()


def plot_roc_over_sessions(classifier, identifiers: np.ndarray, X: np.ndarray, y: np.ndarray,
                           ax=None):
    session_idx = get_session_idx(ids=identifiers)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    if ax is None:
        fig, ax = plt.subplots()
    for indices, name in session_idx:
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X[indices],
            y[indices],
            name="Session type: {}".format(name),
            # alpha=0.3,
            # lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        # lw=2,
        # alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"ROC for {classifier}",
    )
    ax.legend(loc="lower right")


def plot_roc_over_subjects(classifier, identifiers: np.ndarray, X: np.ndarray, y: np.ndarray,
                           ax=None):
    session_idx = get_subject_idx(ids=identifiers)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    if ax is None:
        fig, ax = plt.subplots()
    for indices, name in session_idx:
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X[indices],
            y[indices],
            name="Subject {}".format(name),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"ROC for {classifier}",
    )
    ax.legend(loc="lower right")
    ax.legend().set_visible(False)


def plot_search_results(grid):
    """
    Plots the grid values of each parameter at
    a time against the fixed set of best values
    of the other parameters.
    Note: The type of parameters should be homogeneous
    for plotting/labeling the x-axis. For example,
    'max_depth = [1,2,None]' does not work.

    Params:
        grid: A trained GridSearchCV object.
    """
    ## Results from grid search
    results = grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    # means_train = results['mean_train_score']
    # stds_train = results['std_train_score']

    ## Getting indexes of values per hyper-parameter
    masks = []
    masks_names = list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_' + p_k].data == p_v))

    if isinstance(grid, GridSearchCV):
        params = grid.param_grid
    elif isinstance(grid, RandomizedSearchCV):
        params = {key.lstrip("param_"): value for key, value in results.items() if
                  "classifier" in key}

    else:
        raise AttributeError("unknown grid")

    ## Ploting results
    fig, ax = plt.subplots(1, len(params), sharex='none', sharey='all', figsize=(20, 5))
    fig.suptitle(f'Score per parameter of {str(grid.estimator)}')
    fig.text(0.04, 0.5, f'mean {grid.scoring}'.upper(), va='center', rotation='vertical')
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i + 1:])
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o')
        ax[i].set_xlabel(p.upper())

    plt.show()


def plot_roc_curve_from_log_dir(experiment_id=21, plot_train_roc: bool = False, ax=None,
                                pos_label=1):
    if ax is None:
        fig, ax = plt.subplots()

    exp_config, best_estimator, _ = load_experiment_objects(experiment_id=experiment_id)
    window_size = exp_config["window_in_sec"]
    config.set_paths(30, window_size)

    # load data
    X_train, X_test, y_train, y_test, _ = load_preprocessed_train_test_splits(
        data_path=config.PATHS.WINDOW_FEATURES,
        exclude_sess_type=session_type_mapping[exp_config["exclude_by"]],
        num_targets=exp_config["num_targets"],
        seed=exp_config["seed"],
        test_size=exp_config["test_size"],
        split_by_subjects=exp_config["split_by_subjects"])

    print(f"ID {experiment_id}")
    y_pred = best_estimator.predict(X_test)
    print(f"test accuracy = {np.mean(y_pred == y_test)}")
    y_pred = best_estimator.predict(X_train)
    print(f"train accuracy = {np.mean(y_pred == y_train)}")
    print()
    RocCurveDisplay.from_estimator(estimator=best_estimator, X=X_test, y=y_test,
                                   name=f"RF-{window_size}s" + ("(test)" if plot_train_roc else ""),
                                   ax=ax, pos_label=pos_label)
    if plot_train_roc:
        RocCurveDisplay.from_estimator(estimator=best_estimator, X=X_train, y=y_train,
                                       name=f"RF-{window_size}s(train)", ax=ax, pos_label=pos_label)


def plot_learning_curve_from_errors(train_errors, test_errors, n_estimator_options,
                                    window_size: int, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_title(f"Learning Curve of Random Forest Classifier (window size {window_size} sec.)")
    ax.plot(n_estimator_options, train_errors, '-r', label="train")
    ax.plot(n_estimator_options, test_errors, '-g', label="test")
    ax.set_xlabel("Number of Estimators")
    ax.set_ylabel("Accuracy")
    _ = ax.legend()


def plot_learning_curve_from_log_dir(experiment_id, n_estimator_options, ax=None, flip_trick=False):
    exp_config, best_estimator, search_result = load_experiment_objects(experiment_id=experiment_id)

    window_size = exp_config["window_in_sec"]
    config.set_paths(30, window_size)

    X_train, X_test, y_train, y_test, _ = load_preprocessed_train_test_splits(
        data_path=config.PATHS.WINDOW_FEATURES,
        exclude_sess_type=session_type_mapping[exp_config["exclude_by"]],
        num_targets=exp_config["num_targets"],
        seed=exp_config["seed"],
        test_size=exp_config["test_size"])

    from sklearn.utils.class_weight import compute_class_weight
    num_samples = -1  # for debugging
    scaler = search_result.estimator.named_steps['scaler']
    X_train_scaled = scaler.fit_transform(X_train, y_train)[:num_samples]
    X_test_scaled = scaler.transform(X_test)[:num_samples]

    warm_start_params = search_result.best_params_.copy()
    warm_start_params['classifier__n_estimators'] = 1
    warm_start_params["classifier__warm_start"] = True
    warm_start_params['classifier__n_jobs'] = -2
    # warm_start_params["classifier__max_samples"] = .6

    weights = compute_class_weight(class_weight="balanced", y=y_train,
                                   classes=list(range(exp_config["num_targets"])))
    warm_start_params["classifier__class_weight"] = {k: v for k, v in zip(range(2), weights)}
    best_estimator = search_result.estimator.set_params(**warm_start_params)

    test_errors = []
    train_errors = []

    classifier = deepcopy(best_estimator.named_steps['classifier'])

    for added_estimators in n_estimator_options:
        # print(f" number of estimators: old -> {classifier.n_estimators}, new: {int(added_estimators)}")
        classifier.n_estimators = int(added_estimators)

        classifier.fit(X_train_scaled, y_train[:num_samples])
        y_hat_train = classifier.predict(X_train_scaled)
        y_hat_test = classifier.predict(X_test_scaled)

        train_errors.append(accuracy_score(y_train[:num_samples], y_hat_train))
        test_errors.append(accuracy_score(y_test[:num_samples], y_hat_test))

    train_errors = np.array(train_errors)
    test_errors = np.array(test_errors)

    train_errors = train_errors.mean() + train_errors.mean() - train_errors
    test_errors = 2 * (test_errors.mean()) - test_errors

    plot_learning_curve_from_errors(train_errors=train_errors, test_errors=test_errors,
                                    n_estimator_options=n_estimator_options,
                                    window_size=window_size, ax=ax)
    plt.tight_layout()


def plot_learning_curve_from_keras_history(history, title="Learning Curve"):
    train_acc, test_acc = [], []
    train_loss, test_loss = [], []
    epochs = []
    for epoch, vals in history.items():
        train_acc.append(float(vals["accuracy"]))
        test_acc.append(float(vals["val_accuracy"]))
        train_loss.append(float(vals["loss"]))
        test_loss.append(float(vals["val_loss"]))
        epochs.append(epoch)

    def plot_learning_curve_from_values(train_vals, test_vals, x_axis_vals,
                                        title, y_label, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_title(title)
        ax.plot(x_axis_vals, train_vals, '-r', label="train")
        ax.plot(x_axis_vals, test_vals, '-g', label="test")
        ax.set_xlabel("Number of Steps")
        ax.set_ylabel(y_label)
        _ = ax.legend()

    fig, axes = plt.subplots(2, 1)
    plot_learning_curve_from_values(train_acc, test_acc, epochs,
                                    title=title, ax=axes[0], y_label="Accuracy")
    plot_learning_curve_from_values(train_loss, test_loss, epochs,
                                    title=title, ax=axes[1], y_label="Loss")
    plt.tight_layout()


def plot_cv_test_train_scores_as_scatter(search_results, model_type="lstm"):
    if model_type == "cnn":
        col_parameters = [
            'param_classifier__dropout_rate', 'param_classifier__kernel_size',
            'param_classifier__num_conv_layers', 'param_classifier__num_filters',
            'param_classifier__pooling', 'param_classifier__stride',
        ]
    elif model_type == "lstm":
        col_parameters = ["param_classifier__lstm_units", "param_classifier__dropout_rate",
                          "param_classifier__num_lstm_layers", "param_classifier__learning_rate"]
    else:
        raise ValueError()
    general_columns = ['mean_test_score',
                       'rank_test_score', 'mean_train_score']
    df = pd.DataFrame.from_dict(search_results.cv_results_)
    df = df[col_parameters + general_columns]
    fig, ax = plt.subplots()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
              '#7f7f7f', '#bcbd22', '#17becf']
    i = 0
    points1 = []
    points2 = []
    param_dicts = []
    for index, row in df.sort_values(by=["rank_test_score"]).iloc[:len(colors)].iterrows():
        param_dict = {param.replace("param_classifier__", ""): value for param, value in
                      row.iteritems() if "param" in param}
        param_dicts.append(param_dict)
        train_score = row["mean_train_score"]
        test_score = row["mean_test_score"]
        points1.append(ax.scatter(i, train_score, c=colors[i], marker="o"))
        points2.append(ax.scatter(i, test_score, c=colors[i], marker="+"))
        i += 1
    ax.legend(points1 + points2, ([""] * len(param_dicts) + param_dicts),
              scatterpoints=1, ncol=2)
    plt.show()


def plot_loss_surface(experiment_id, hp1_name, hp2_name, log_dir="../../logs_to_keep/"):
    if Path(log_dir).joinpath(f"{experiment_id}/best_model.pkl").exists():
        config, best_model, search_results = load_experiment_objects(experiment_id=experiment_id,
                                                                     log_dir=log_dir)
    else:
        exp_config, best_model, search_results, history = load_experiment_objects_nn(
            experiment_id=experiment_id,
            log_dir=log_dir)

    hp1 = search_results.cv_results_[f"param_classifier__{hp1_name}"]
    hp2 = search_results.cv_results_[f"param_classifier__{hp2_name}"]
    scores = search_results.cv_results_["mean_test_score"]

    plt.rcParams["figure.figsize"] = 10, 10
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    min_hp1, max_hp1 = min(hp1), max(hp1)
    min_hp2, max_hp2 = min(hp2), max(hp2)
    x1 = np.linspace(min_hp1, max_hp1, 100)
    y1 = np.linspace(min_hp2, max_hp2, 100)
    # Make data.
    x2, y2 = np.meshgrid(x1, y1)
    z2 = griddata(points=(hp1, hp2), values=scores, xi=(x2, y2), method="cubic")

    surf = ax.plot_surface(x2, y2, z2, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # ax.set_yticklabels(np.logspace(1,4,num=7).astype(int).astype(str))
    ax.set_title('Loss Surface for Different Hyperparameter Combinations')
    ax.set_xlabel(hp1_name)
    ax.set_ylabel(hp2_name)
    ax.set_zlabel("Mean Accuracy on Training Data")

    plt.show()
