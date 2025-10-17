# -*- coding: utf-8 -*-
# Author: Volkan Erol (Marmara University)
# Contact: volkan.erol@gmail.com
# License: MIT

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def build_feature_map(x, reps=2):
    """Build a 6-qubit variational feature map with RY/RZ encoding and CZ entanglement."""
    qc = QuantumCircuit(6)
    for _ in range(reps):
        for i in range(6):
            angle = float(x[i % len(x)])
            qc.ry(angle, i)
            qc.rz(angle, i)
        for i in range(5):
            qc.cz(i, i+1)
    return qc

def quantum_kernel(x1, x2):
    """Compute |⟨ψ(x1)|ψ(x2)⟩|² quantum kernel."""
    qc1 = build_feature_map(x1)
    qc2 = build_feature_map(x2)
    psi1 = Statevector.from_instruction(qc1)
    psi2 = Statevector.from_instruction(qc2)
    return np.abs(psi1.data.conj().dot(psi2.data))**2
