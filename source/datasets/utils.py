import enum


class DatasetMode(enum.StrEnum):
    # Should be exactly same with subsets in configuration file.
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
