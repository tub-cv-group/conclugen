from utils import constants as C


def fer_label_for_class_index(idx: int):
    """Returns the label for index idx of the default label set (see constants.py).

    Args:
        idx (int): the index to convert to a string label

    Returns:
        _type_: the string label
    """
    return C.CAER_EXPRESSION_LABELS[idx]


def class_index_for_fer_label(label: str):
    """Returns the index of the given label in the default label set (see constants.py).

    Args:
        label (str): the label to return the index for

    Raises:
        Exception: if the label is not in the label set

    Returns:
        _type_: the index of the label in the default label set
    """
    if label in C.CAER_EXPRESSION_LABELS:
        return C.CAER_EXPRESSION_LABELS.index(label)
    else:
        if label == 'Angry':
            return 0
        else:
            raise Exception(f'Label \'{label}\' is not supported.')
