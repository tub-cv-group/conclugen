from typing import List, Union
from torchvision.transforms import RandomHorizontalFlip, Compose


def transforms_list_contains_type(transforms: Union[List, Compose], transform_class: type, pop: bool=True):
    """Checks the transforms list for the given type and returns it if found.
    If pop is set to True, the respective element will also be removed from the
    list.

    Args:
        transforms (List): the list to check
        transform_class (type): the type to check for
        pop (bool, optional): If set to true, removes the respective element.
        Defaults to True.

    Returns:
        object: The transform if found, None otherwise.
    """
    if isinstance(transforms, Compose):
        # access the interal transforms
        transforms = transforms.transforms
    for i in range(len(transforms)):
        transform = transforms[i]
        if isinstance(transform, transform_class):
            if pop:
                transforms.remove(transform)
            return transform
    return None