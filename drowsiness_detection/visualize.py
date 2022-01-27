import pandas as pd

pd.set_option("display.precision", 2)
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_cm_matrix(model, X_test, y_test):
    y_pred = model.predict_classes(X_test)
    y_test_single_digit = np.argmax(y_test, axis=-1)
    c_matrix = confusion_matrix(y_test_single_digit, y_pred)
    plt.figure()
    plot_confusion_matrix(c_matrix, classes=list("0123456789"))


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
        line.set_data(np.arange(n - n_frames // 2, n + n_frames // 2), data[n - n_frames // 2:n + n_frames // 2])
        txt_title.set_text(f"Frame = {n}")
        ax.set_xlim((n - n_frames // 2, n + n_frames // 2))
        return line,

    anim = animation.FuncAnimation(fig, drawframe, frames=range(n_frames // 2, len(data) - n_frames // 2), interval=20, blit=True)
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

    allowed_frames = np.arange(n_frames // 2, len(data) - n_frames // 2, step=2)  # does the video frame start at 0 or 1?

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
