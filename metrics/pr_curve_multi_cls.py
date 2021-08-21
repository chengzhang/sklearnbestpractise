# coding = utf8

from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
import logging
import numpy as np
import matplotlib.pyplot as plt


def plot_pr_curve_multi_cls(y_true, y_pred, class_names=None, show=True, out_fig_file=None):
    """
    :param y_true: list of int. shape: [n_sample]. the golden category of each sample
    :param y_pred: list of list of float. shape: [n_sample, n_class]. category prediction scores of each sample
    :param class_names: name of class
    :param show: show the figure directly.
    :param out_fig_file: save the figure into a file.
    :return: no return
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        logging.fatal('invaild len of y_ture or y_pred')
    n_class = len(y_pred[0])
    class_indexes = list(range(n_class))
    y_one_hot = label_binarize(y_true, classes=class_indexes)
    y_pred = np.asarray(y_pred)
    precision_recall_thresholds = []
    average_precisions = []
    for i in range(n_class):
        precision_recall_thresholds.append(
            precision_recall_curve(y_one_hot[:, i], y_pred[:, i])
        )
        average_precisions.append(average_precision_score(y_one_hot[:, i], y_pred[:, i]))

    micro_precision, micro_recall, micro_threshold = precision_recall_curve(y_one_hot.ravel(), y_pred.ravel())
    micro_ap = average_precision_score(y_one_hot, y_pred, average='micro')

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    plt.figure(figsize=(20, 12))
    lines = []
    labels = []
    l, = plt.plot(micro_recall, micro_precision, lw=2)
    lines.append(l)
    labels.append('micro-average precision-recall(area = {0:0.3f})'.format(micro_ap))
    for i in range(n_class):
        l, = plt.plot(precision_recall_thresholds[i][1], precision_recall_thresholds[i][0])
        lines.append(l)
        name = class_names[i] if class_names else i
        labels.append('class {}'.format(name) + ' precision-recall(area = {0:0.3f})'.format(average_precisions[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.40)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    if show:
        plt.show()
    if out_fig_file:
        plt.savefig(out_fig_file)
