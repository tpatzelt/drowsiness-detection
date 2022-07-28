from pathlib import Path

SOURCES_ROOT_PATH = Path(__file__).parent
DATA_DIR_PATH = SOURCES_ROOT_PATH.parent.joinpath("data")
MODEL_DIR_PATH = DATA_DIR_PATH.joinpath("models/60sec")
PREDICTION_DIR_PATH = DATA_DIR_PATH.joinpath("predictions/60sec")

BASE_PATH = Path("/home/tim/Windows")
# BASE_PATH = Path("/media/tim/My Passport/drowsiness_data/Windows")
PATHS = None


class Paths:
    """Object holding all data paths."""

    def __init__(self, frequency: int, seconds: int, label_string: str = "KSS"):
        self.frequency = frequency
        frequency_string = str(frequency) + "_Hz"
        self.seconds = seconds
        seconds_string = str(seconds) + "_sec"
        self.DATA = BASE_PATH.joinpath("Windows_" + frequency_string)
        self.WINDOW_DATA = self.DATA.joinpath("WindowData").joinpath(seconds_string)
        self.LABEL_DATA = self.DATA.joinpath("WindowLabels").joinpath(label_string).joinpath(
            seconds_string)
        self.TRAIN_TEST_SPLIT = self.DATA.joinpath("TrainTestSplits").joinpath(seconds_string)
        self.SPLIT_IDENTIFIER = self.TRAIN_TEST_SPLIT.joinpath("identifiers")

        self.DATA_FORMAT_FILE = self.DATA.joinpath("WindowData").joinpath("Format").joinpath(
            "data_format.json")
        self.LABEL_FORMAT_FILE = self.DATA.joinpath("Format").joinpath("kss_Label_format.json")
        self.FEATURE_NAMES_FILE = self.DATA.joinpath("WindowFeatures").joinpath(
            "feature_names.txt")

        self.WINDOW_FEATURES = self.DATA.joinpath("WindowFeatures").joinpath(seconds_string)


def set_paths(frequency: int, seconds: int):
    "Switch function to change data sources."
    global PATHS
    PATHS = Paths(frequency=frequency, seconds=seconds)


set_paths(1, 1)
