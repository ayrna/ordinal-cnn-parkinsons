"""Evaluation metrics

A compendium of evaluation metrics used in the experimentation
"""

import warnings
import numpy as np

from sklearn.metrics import confusion_matrix

import scipy.stats


# Correct Classification Rate
def ccr(y, ypred):
    with warnings.catch_warnings():
        return (y == ypred).sum() / len(y)


# Geometric Mean of the Sensitivities (GMS)
def gm(y, ypred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = confusion_matrix(y, ypred)
        sum_byclass = np.sum(cm, axis=1)
        sensitivities = np.diag(cm) / sum_byclass.astype('double')
        geometric_mean = np.prod(sensitivities) ** (1.0 / cm.shape[0])
        return geometric_mean


# Minimum Sensitivity (MS)
def ms(y, ypred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = confusion_matrix(y, ypred).astype(float)
        sum_byclass = np.sum(cm, axis=1).astype(float)
        sensitivities = np.diag(cm) / sum_byclass
        ms = np.min(sensitivities)

        return ms


# Mean Absolute Error (MAE)
def mae(y, ypred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = np.asarray(y)
        ypred = np.asarray(ypred)
        return abs(y - ypred).sum() / len(y)


# Average MAE
def amae(y, ypred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = confusion_matrix(y, ypred)
        n_class = cm.shape[0]
        costes = np.reshape(np.tile(list(range(n_class)), n_class), (n_class, n_class))
        costes = np.abs(costes - np.transpose(costes))
        errores = costes * cm
        amaes = np.sum(errores, axis=1) / np.sum(cm, axis=1).astype('double')
        amaes = amaes[~np.isnan(amaes)]
        return np.mean(amaes)


# Maximum MAE
def mmae(y, ypred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = confusion_matrix(y, ypred)
        n_class = cm.shape[0]
        costes = np.reshape(np.tile(list(range(n_class)), n_class), (n_class, n_class))
        costes = np.abs(costes - np.transpose(costes))
        errores = costes * cm
        amaes = np.sum(errores, axis=1) / np.sum(cm, axis=1).astype('double')
        amaes = amaes[~np.isnan(amaes)]
        return amaes.max()


# Weighted Cohen's Kappa (linear weighing)
def wkappa(y, ypred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        cm = confusion_matrix(y, ypred)
        n_class = cm.shape[0]
        costes = np.reshape(np.tile(list(range(n_class)), n_class), (n_class, n_class))
        costes = np.abs(costes - np.transpose(costes))
        f = 1 - costes

        n = cm.sum()
        x = cm / n

        r = x.sum(axis=1)  # Row sum
        s = x.sum(axis=0)  # Col sum
        Ex = r.reshape(-1, 1) * s
        po = (x * f).sum()
        pe = (Ex * f).sum()
        return (po - pe) / (1 - pe)


# Kendall rank correlation coefficient (Kendall's tau)
def tkendall(y, ypred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        corr, pvalue = scipy.stats.kendalltau(y, ypred)

        if np.isnan(corr):
            return 0.0
        else:
            return corr


# Spearman rank correlation coefficient
def spearman(y, ypred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        n = len(y)
        num = ((y - np.repeat(np.mean(y), n)) * (ypred - np.repeat(np.mean(ypred), n))).sum()
        div = np.sqrt(
            (pow(y - np.repeat(np.mean(y), n), 2)).sum() * (pow(ypred - np.repeat(np.mean(ypred), n), 2)).sum())

        if num == 0:
            return 0
        else:
            return num / div


metric_list = [
    ccr,
    amae,
    gm,
    mae,
    mmae,
    ms,
    tkendall,
    wkappa,
    spearman,
]
