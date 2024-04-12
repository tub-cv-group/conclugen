from typing import List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd


def save_confusion_matrix(
    matrix: np.array,
    ylabels: List[str],
    xlabels: List[str],
    file_path: str,
    fmt: str='d',
    ax: plt.Axes=None
):
    """Plots the given matrix as a confusion matrix using plot_confusion_matrix
    and saves the resulting figure at the specified path.

    Args:
        matrix (np.array): the matrix to plot as confusion matrix
        ylabels (List[str]): the labels for the x axis
        xlabels (List[str]): the labels for the y axis
        file_path (str): the path where to store the resulting figure
        ax (plt.Axes, optional): the axes to reuse. Defaults to None.
    """
    plot_confusion_matrix(matrix, ylabels, xlabels, fmt, ax)
    confusion_matrix_path = file_path
    plt.savefig(confusion_matrix_path)
    plt.clf()

def plot_confusion_matrix(
    matrix: np.array,
    xlabels: List[str],
    ylabels: List[str],
    fmt: str='d',
    ax: plt.Axes=None,
    vmin: float=None,
    vmax: float=None
) -> plt.Axes:
    """Plots the given matrix as a confusion matrix while using xlabels and
    ylabels to label the axes. This function internally uses seaborn and returns
    the axes of the resulting figures (i.e. does not save the figure).

    Args:
        matrix (np.array): the matrix to plot as a confusion matrix
        xlabels (List[str]): the labels for the x
        ylabels (List[str]): the labels for the y axis
        fmt (str, optional): to format the matrix entries (d is digit). Defaults to 'd'.
        ax (matplotlib.pyplot.Axes, optional): the axes to reuse. Defaults to None.
        vmin (float, optional): minimum value. Defaults to None.
        vmax (float, optional): maximum value. Defaults to None.

    Returns:
        mstplotlib.pyplot.axes: the axes of the figure, either the passed ones or new ones
    """
    df_cm = pd.DataFrame(
        matrix,
        range(len(ylabels)),
        range(len(xlabels)))
    sn.set(font_scale=1.0)  # for label size
    ax = sn.heatmap(df_cm,
                    annot=True,
                    cmap='Blues',
                    annot_kws={"size": 8},
                    fmt=fmt,
                    xticklabels=xlabels,
                    yticklabels=ylabels,
                    ax=ax,
                    vmin=vmin,
                    vmax=vmax)  # font size
    ax.figure.tight_layout()
    return ax