# -*- coding: utf-8 -*-
# Author: Volkan Erol
# Classical baseline implementations

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

def train_baselines(X_train, y_train):
    models = {
        'SVR': SVR(kernel='rbf', C=10, gamma=0.1),
        'RF': RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=4),
        'GBM': GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=6),
        'NN': MLPRegressor(hidden_layer_sizes=(64, 32, 16), max_iter=300, random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models
