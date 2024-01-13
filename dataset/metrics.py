from enum import Enum


class Metric(Enum):
    EUCLIDEAN = 1
    DOT_PRODUCT = 2
    ANGULAR = 3
    JACCARD = 4
    HAMMING = 5
