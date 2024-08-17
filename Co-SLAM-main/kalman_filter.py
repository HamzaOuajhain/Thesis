import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, initial_P, F, H, Q, R):
        self.state = initial_state
        self.P = initial_P
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
    
    def predict(self):
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
    
    def update(self, measurement):
        y = measurement - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state = self.state + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)