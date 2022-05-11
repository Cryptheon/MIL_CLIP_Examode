import numpy as np
import pandas as pd
from argparse import ArgumentParser
import os

def precision_at_k(ordered_array, k):
    precision = ordered_array[:k+1].mean()
    return precision

def AP(ordered_array):
    summed = []
    for i in range(ordered_array.shape[0]):
        p = precision_at_k(ordered_array, i)
        indicator = ordered_array[i]
        summed.append(p*indicator)

    ap = np.array(summed).sum() / ordered_array.sum()
    return ap

def binarize(multi_labels):
    #carcinoma
    if multi_labels[0]==1:
        return 1
    #HGD
    elif multi_labels[1]==1:
        return 1
    #LGD
    elif multi_labels[2]==1:
        return 1
    # else just normal tissue
    return 0

def compute_metrics(sorted_multi_labels):
    sorted_binary_labels = [binarize(labels) for labels in sorted_multi_labels]
    ap = AP(np.array(sorted_binary_labels))
    return ap, sorted_binary_labels
