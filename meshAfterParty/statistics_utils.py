import numpy as np

def precision(TP,FP):
    if (TP + FP) > 0:
        return TP/(TP + FP)
    else:
        return np.nan
    
def recall(TP,FN):
    if (TP + FN) > 0:
        return TP/(TP + FN)
    else:
        return np.nan
    
def f1(TP,FP,FN):
    curr_prec = precision(TP,FP)
    curr_recall = recall(TP,FN)
    
    if curr_prec + curr_recall > 0:
        return 2*(curr_prec*curr_recall)/(curr_prec + curr_recall)
    else:
        return np.nan
    
def calculate_scores(TP,FP,FN):
    return dict(precision=precision(TP,FP),
               recall=recall(TP,FN),
               f1=f1(TP,FP,FN))