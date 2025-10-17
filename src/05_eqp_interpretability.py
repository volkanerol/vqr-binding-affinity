# -*- coding: utf-8 -*-
# Author: Volkan Erol
# Explainable Quantum Pharmacology (EQP) sensitivity analysis

import numpy as np

def parameter_shift_gradients(kernel_fn, x, x_ref, eps=np.pi/2):
    grads = []
    for j in range(len(x)):
        x_fwd, x_bwd = x.copy(), x.copy()
        x_fwd[j] += eps
        x_bwd[j] -= eps
        grad = 0.5 * (kernel_fn(x_fwd, x_ref) - kernel_fn(x_bwd, x_ref))
        grads.append(grad)
    return np.array(grads)

def compute_feature_importance(X, y, alpha, kernel_fn):
    importance = np.zeros(X.shape[1])
    for i, x in enumerate(X):
        grad = parameter_shift_gradients(kernel_fn, x, X[i])
        importance += np.abs(alpha[i]) * np.abs(grad)
    importance /= np.sum(importance)
    return importance
