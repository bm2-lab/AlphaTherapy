import numpy as np
from sklearn import metrics

def sample_based_pcc(y_true, y_pred, low_gene_thre, high_gene_thre):
    """pcc : calculate pearson correlation coefficient average by gene

    Parameters
    ----------
    y_true : np.ndarray [n*m]
        true label
    y_pred : np.ndarray [n*m]
        predict label
    gene_thre : np.ndarray [m]
        label thresholds
        which we use to select the significant expression values

    Returns
    -------
    list
        [pcc_value, pcc_value_precision]
    """

    pcc_value = 0.0
    pcc_value_precision = 0.0

    sample_number = y_pred.shape[0]
    precision_null_sample_number = 0

    original_pcc_arr = np.zeros(sample_number)
    precision_pcc_arr = np.zeros(sample_number)

    # calculate PCC by sample
    for i in range(sample_number):

        flag = False

        # 1. standard pcc
        original_pcc_arr[i] = np.corrcoef(y_true[i, :], y_pred[i, :])[0, 1]
        pcc_value += np.corrcoef(y_true[i, :], y_pred[i, :])[0, 1]

        # 2. precision pcc(focusing on true data)
        a = y_true[i, :].copy()
        b = y_pred[i, :].copy()
        low_gene_thre_val, high_gene_thre_val = np.percentile(np.abs(a), [low_gene_thre, high_gene_thre])
        ind = (np.abs(a) > low_gene_thre_val) & (np.abs(a) <= high_gene_thre_val)
        if np.sum(ind) > 1:
            a = a[ind]
            b = b[ind]
            temp1 = np.corrcoef(a, b)[0, 1]
            pcc_value_precision += temp1
            precision_pcc_arr[i] = temp1
        else:
            precision_null_sample_number += 1
            flag = True

    return [original_pcc_arr, precision_pcc_arr]
