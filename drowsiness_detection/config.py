from pathlib import Path

# DATA_PATH = Path(__file__).parent.parent.joinpath("data/")
# PREPROCESSED_DATA_PATH = DATA_PATH.joinpath("preprocessed")

DATA_PATH = Path("/home/tim/Windows/")
WINDOW_DATA_PATH = DATA_PATH.joinpath("WindowData")
WINDOW_LABEL_PATH = DATA_PATH.joinpath("WindowLabels")
WINDOW_FEATURES_PATH = DATA_PATH.joinpath("WindowFeatures")
DATA_FORMAT_PATH = DATA_PATH.joinpath("data_format.json")
LABEL_FORMAT_PATH = DATA_PATH.joinpath("kss_labels_format.json")
FEATURE_NAMES_PATH = DATA_PATH.joinpath("features_names.txt")
