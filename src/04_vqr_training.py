# -*- coding: utf-8 -*-
# Author: Volkan Erol
# Variational Quantum Regression training pipeline

import numpy as np
from qiskit import Aer, execute
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn.kernel_ridge import KernelRidge

from .quantum_kernel import build_feature_map

def train_vqr(X_train, y_train, X_test, y_test, shots=1024):
    backend = Aer.get_backend('aer_simulator_statevector')
    qkernel = QuantumKernel(feature_map=build_feature_map, quantum_instance=QuantumInstance(backend))
    K_train = qkernel.evaluate(X_train)
    K_test = qkernel.evaluate(X_test, X_train)
    model = KernelRidge(alpha=0.01, kernel="precomputed")
    model.fit(K_train, y_train)
    preds = model.predict(K_test)
    r2 = 1 - np.sum((preds - y_test)**2) / np.sum((y_test - np.mean(y_test))**2)
    print(f"Test R²: {r2:.3f}")
    return model, preds
