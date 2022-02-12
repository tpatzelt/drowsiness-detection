import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.widgets import Slider
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold

from drowsiness_detection.data import get_session_idx, get_subject_idx


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


def plot_roc_over_n_folds(classifier, X, y, n_splits=6, fit_model=False):
    cv = StratifiedKFold(n_splits=n_splits)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

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
        title="Receiver operating characteristic example",
    )
    ax.legend(loc="lower right")
    plt.show()


def plot_roc_over_sessions(classifier, identifiers: np.ndarray, X: np.ndarray, y: np.ndarray):
    session_idx = get_session_idx(ids=identifiers)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for indices, name in session_idx:
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X[indices],
            y[indices],
            name="Session type: {}".format(name),
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
        title="Receiver operating characteristic example",
    )
    ax.legend(loc="lower right")
    plt.show()


def plot_roc_over_subjects(classifier, identifiers: np.ndarray, X: np.ndarray, y: np.ndarray):
    session_idx = get_subject_idx(ids=identifiers)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

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
        title="Receiver operating characteristic example",
    )
    ax.legend(loc="lower right")
    ax.legend().set_visible(False)
    plt.show()


def plot_search_results(grid):
    """
    Plots the grid values of each parameter at
    a time against the fixed set of best values
    of the other parameters.
    Note: The type of parameters should be uniform
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

    params = grid.param_grid

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
