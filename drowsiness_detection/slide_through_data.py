import click
import matplotlib.pyplot as plt

from drowsiness_detection.data import create_eye_closure_karolinksa_dataset
from drowsiness_detection.visualize import show_frame_slider


@click.command()
@click.option("--session", default="001_1_a", help="session id")
@click.option("--window", default=120, help="frame windows shown.")
def slide_through_eye_closure_data(session: str, window: int):
    gen = create_eye_closure_karolinksa_dataset()
    session_data = None
    for data in gen:
        if data.name == session:
            session_data = data
            break
    if not session:
        raise ValueError(f"Did not find session {session}.")

    slider = show_frame_slider(data=session_data["eye_closure"], n_frames=window)
    plt.show(block=True)


if __name__ == '__main__':
    slide_through_eye_closure_data()
