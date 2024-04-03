import numpy as np

class Communication:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.state = initial_state  # Initial state estimate
        self.covariance = initial_covariance  # Initial covariance estimate
        self.process_noise = process_noise  # Process noise covariance matrix
        self.measurement_noise = measurement_noise  # Measurement noise covariance matrix
