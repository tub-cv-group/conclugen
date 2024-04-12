from pathlib import Path
import shutil
from typing import List
import os
import glob
from natsort import natsorted
from pathlib import Path
import pathlib
import hashlib
import mlflow
from mlflow.entities import RunInfo
from pytorch_lightning.loggers.mlflow import LOCAL_FILE_URI_PREFIX


def remove_extension_from_path(path: str):
    path_with_extension_removed = os.path.splitext(path)[0]
    return path_with_extension_removed


def get_extension(path: str):
    ext = os.path.splitext(path)[1]
    return ext


def get_filename_without_extension(path: str):
    """Returns the filename of the given path without the extension.

    Args:
        path (str): the path to return without extension

    Returns:
        str: the path without extension
    """
    filename_without_extension = os.path.splitext(os.path.basename(path))[0]
    return filename_without_extension


def get_parent_dir(path: str):
    parent_dir = os.path.basename(os.path.dirname(path))
    return parent_dir


def sorted_listdir(path: str, filter: str=None) -> List:
    """Returns os.listdir(path) but sorted naturally using natsort.

    Args:
        path (str): the path to list and sort
        filter (str): to filter the sorted list. Put 'dir' to filter for dirs
            or a file extension to return only files of that type.

    Returns:
        List: the sorted os.listdir
    """
    sorted_list = natsorted(os.listdir(path))
    if filter:
        if filter == 'dir':
            sorted_list = [entry for entry in sorted_list if os.path.isdir(os.path.join(path, entry))]
        else:
            sorted_list = [entry for entry in sorted_list if entry.endswith(filter)]
    return sorted_list


def paths_append_and_recreate(paths: List, to_append: List):
    """Appends to each path in paths the string at the same index in to_append,
    checks if the path exists and, if yes, removes it and recreates it.

    Args:
        paths (List): the paths to use as prefix
        to_append (List): the paths to append

    Returns:
        List: to appended paths in same order
    """
    to_return = []
    for idx, path in enumerate(paths):
        appended_path = os.path.join(
            path,
            to_append[idx]
        )
        to_return.append(appended_path)
        if os.path.exists(appended_path):
            shutil.rmtree(appended_path)
        os.makedirs(appended_path)
    return to_return


def num_files(data_dir: str, ext: str):
    """Returns the number of files found recursively by glob.glob with the
    specified extension.

    Args:
        data_dir (str): the root datadir
        ext (str, optional): the extension of the images. If None, any files will be matched with the wild card *.*.

    Returns:
        int: the number of files found
    """
    if ext is not None:
        img_files = glob.glob(os.path.join(data_dir, '**', f'*.{ext}'), recursive=True)
    else:
        img_files = glob.glob(os.path.join(data_dir, '**', '*.*'), recursive=True)
    return len(img_files)


def num_files_ok(data_dir: str, expected: int, ext: str):
    """Checks whether the number of images found recursively by glob.glob with
    the specified extension matches the expected number of images. It prints
    a warning if it doesn't match.

    Args:
        data_dir (str): the root datadir
        expected (int): the number of expected images
        ext (str, optional): the extension of the images. If None, any files will be matched with the wild card *.*.

    Returns:
        _type_: true, if the number matches the expected number
    """
    found = num_files(data_dir, ext)
    is_ok = found == expected
    if not is_ok:
        print(f'Number of files in dir {data_dir} not correct '
              f'(expected {expected}, found {found}).')
    return is_ok


def filesize_ok(filepath: str, expected: int, remove_if_corrput: bool = False) -> bool:
        """Verifies the size of the text file containing the extracted subtitles
        of the respective subset (e.g. train or test).

        Args:
            filepath (str): the path of the file to test
            expected (int): the expected file size
            remove_if_corrput (bool, optional): whether to remove the file if
            corrupt. Defaults to False.

        Returns:
            bool: whether the file has the right size
        """
        if not os.path.exists(filepath):
            print(f'File {filepath} does not exist but should.')
            return False
        actual_filesize = os.stat(filepath).st_size
        if actual_filesize != expected:
            if remove_if_corrput and os.path.exists(filepath):
                print(f'File {filepath} is corrupt. Expected size {expected}, got {actual_filesize}. Removing.')
                os.remove(filepath)
            else:
                print(f'File {filepath} is corrupt. Expected size {expected}, got {actual_filesize}.')
            return False
        return True


def rmdir(directory: str):
    """Recursively removes the given directory, all its subdirectories and the
    containing files.
    
    NOTE: Taken from https://stackoverflow.com/a/49782093/3165451.

    Args:
        directory (str): the directory to remove recursively
    """
    directory = Path(directory)
    if not directory.exists():
        return

    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()


def resolve_mlflow_run_dir(run_id: str) -> str:
    """Returns the directory of the run belonging to the given run_id.

    Args:
        run_id (str): the ID of the run whose directory is to be returned

    Returns:
        str: the directory of the run with the given run_id
    """
    run = mlflow.get_run(run_id)
    run_info: RunInfo = run.info
    run_dir = pathlib.Path(run_info.artifact_uri).parent
    run_dir = str(run_dir)
    if LOCAL_FILE_URI_PREFIX in run_dir:
        run_dir = run_dir.lstrip(LOCAL_FILE_URI_PREFIX)
    return run_dir


def sanitize_mlflow_dir(dir: str) -> str:
    """MLflow prepends file:/ to local URIs making them inusable directly. This
    function removes this prefix.

    Args:
        dir (str): the directory to sanitize

    Returns:
        str: the sanitized directory
    """
    if LOCAL_FILE_URI_PREFIX in dir:
        dir = dir.lstrip(LOCAL_FILE_URI_PREFIX)
        if dir.startswith('//'):
            dir = dir[2:]
    return dir


def get_mlflow_artifact_path(run_id: str, artifact_name: str) -> str:
    """Returns the path to the artifact with name artifact_name of the run with
    the given ID. Sanitizes the path from file:// prefix.

    Args:
        run_id (str): the ID of the run which the artifact was logged to
        artifact_name (str): the name of the artifact

    Returns:
        str: the full path to the artifact without file:// prefix
    """
    run_id = mlflow.get_run(run_id)
    artifact_path = os.path.join(
        run_id.info.artifact_uri,
        artifact_name)
    return sanitize_mlflow_dir(artifact_path)


# https://stackoverflow.com/a/4248689/3165451
def parse_range(astr):
    result = set()
    for part in astr.split(','):
        x = part.split('-')
        result.update(range(int(x[0]), int(x[-1])+1))
    return sorted(result)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


def has_file_allowed_extension(filename: str, extensions: str) -> bool:
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def split_path(path_to_split: str) -> List[str]:
    """Split the given path path_to_split into its individual components and
    returns them as a list.

    Args:
        path_to_split (str): the path to split

    Returns:
        list: the invidiual components of the path
    """
    path_list   = []
    while os.path.basename(path_to_split):
        path_list.append( os.path.basename(path_to_split) )
        path_to_split = os.path.dirname(path_to_split)
    path_list.reverse()
    return path_list

def file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()