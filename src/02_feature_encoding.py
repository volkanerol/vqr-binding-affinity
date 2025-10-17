# -*- coding: utf-8 -*-
# Author: Volkan Erol
# Feature encoding and normalization

import numpy as np
from sklearn.preprocessing import StandardScaler

def encode_features(df, descriptor_cols):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[descriptor_cols])
    return X, scaler
