import os
from typing import List

from utils import file_util


def diff(list_a: List, list_b: List) -> List:
    """Computes the difference between the two lists.

    Args:
        listA (List): List a of the diff.
        listB (List): List b of the diff.

    Returns:
        List: The difference between List a and List a
    """
    return list(set(list_a) - set(list_b) | set(list_b) - set(list_a))


def string_repr_to_list_of_strings(string_repr_of_list_of_strings: str) -> List[str]:
    """Takes the string representation of a list of strings and returns 
    a parsed list of strings back.

    Args:
        string_repr_of_list_of_strings (str): the list to parse

    Returns:
        List[str]: the parsed list containing the strings
    """
    # This special case will otherwise be returned as ['']
    if string_repr_of_list_of_strings == '[]':
        return []
    return [i.strip() for i in string_repr_of_list_of_strings[1:-1].replace('\"', '').split(',')]


def intersect_file_paths_based_on_filename(list1: List[str], list2: List[str]):
    """Intersects the file paths based on the filename. To this end, the filename
    without extension is used to compare the two lists, and only the items
    which are present in both are kept.

    For Example:
    list1 = ['a/b/c/d.txt', 'a/b/c/e.txt', 'a/b/c/f.txt']
    list2 = ['a/b/c/d.npy', 'a/b/c/e.npy', 'a/b/c/g.npy']
    intersect_file_paths_based_on_filename(list1, list2) =
        ['a/b/c/d.txt', 'a/b/c/e.txt'], ['a/b/c/d.npy', 'a/b/c/e.npy']

    Args:
        list1 (List[str]): first list of paths
        list2 (List[str]): second list of paths

    Returns:
        Tuple[List[str], List[str]]: the intersected lists
    """
    
    # Create sets with file names (without extension) from each list
    set1 = set(file_util.get_filename_without_extension(path) for path in list1)
    set2 = set(file_util.get_filename_without_extension(path) for path in list2)

    # Get the intersection of the two sets
    intersection = set1 & set2

    # Create lists with file paths from each original list that are in the intersection
    intersected_list1 = [path for path in list1 if file_util.get_filename_without_extension(path) in intersection]
    intersected_list2 = [path for path in list2 if file_util.get_filename_without_extension(path) in intersection]

    return intersected_list1, intersected_list2