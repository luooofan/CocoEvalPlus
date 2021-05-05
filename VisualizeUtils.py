# Reference: https://www.osgeo.cn/matplotlib/gallery/images_contours_and_fields/image_annotated_heatmap.html

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def drawCatEvalBar(data, label, savepath=None):
    # tick_label = ['total', 'car', 'person', 'bus', 'motorbike', 'bicycle']
    tick_label = label
    indexs = ['0.50:0.95', '0.50', '0.75', 'small', 'medium', 'large'] * 2
    data = pd.DataFrame(data, columns=tick_label, index=indexs)

    # draw four multi-classes bar chart
    # left chart represent mAP/mAR of 6 classes(total + 5 class) in the cases of IoU equal to 0.5:0.95, 0.5, 0.75
    # right chart represent mAP/mAR of 6 classes(total + 5 class) in the cases of area equal to 'small','medium','large'
    n = 3
    x = np.arange(data.shape[1])
    total_width = 0.8
    width = total_width / n
    fig, ax = plt.subplots(2, 2, figsize=(18, 10))
    labels = [['0.5:0.95', '0.5', '0.75'], ['small', 'medium', 'large']]
    ax[0, 0].set_ylabel('AP')
    ax[1, 0].set_ylabel('AR')
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            l1 = ax[i, j].bar(x-width, data.iloc[(i+j)*n], fc='b', width=width)
            l2 = ax[i, j].bar(x, data.iloc[(i+j)*n+1], tick_label=tick_label, width=width)
            l3 = ax[i, j].bar(x+width, data.iloc[(i+j)*n+2], width=width)
            new_x = ax[i, j].get_xlim()
            ax[i, j].plot(new_x, [data.iloc[(i+j)*n][0]]*len(new_x), alpha=0.5, linestyle='--', marker='.')
            ax[i, j].plot(new_x, [data.iloc[(i+j)*n+1][0]]*len(new_x), alpha=0.5, linestyle='--', marker='.')
            ax[i, j].plot(new_x, [data.iloc[(i+j)*n+2][0]]*len(new_x), alpha=0.5, linestyle='--', marker='.')
            ax[i, j].legend(handles=[l1, l2, l3], labels=labels[j], loc='best')
            ax[i, j].set_ylim((0, 1))
            ax[i, j].set_yticks(np.arange(0, 1.1, 0.1))
    plt.show()
    if savepath is not None:
        plt.savefig(savepath)
        print('Saved catgory_evaluation_figure!')
