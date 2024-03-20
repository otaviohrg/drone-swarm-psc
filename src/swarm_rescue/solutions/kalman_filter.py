import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.state = initial_state  # Initial state estimate
        self.covariance = initial_covariance  # Initial covariance estimate
        self.process_noise = process_noise  # Process noise covariance matrix
        self.measurement_noise = measurement_noise  # Measurement noise covariance matrix

    def predict(self, A, B, u):
        # Predict state and covariance
        self.state = np.dot(A, self.state) + np.dot(B, u)
        self.covariance = np.dot(np.dot(A, self.covariance), A.T) + self.process_noise

    def update(self, z, H):
        # Update state and covariance based on measurement
        predicted_measurement = np.dot(H, self.state)  # Calculate predicted measurement
        innovation = z - predicted_measurement
        innovation_covariance = np.dot(np.dot(H, self.covariance), H.T) + self.measurement_noise
        kalman_gain = np.dot(np.dot(self.covariance, H.T), np.linalg.inv(innovation_covariance))
        self.state = self.state + np.dot(kalman_gain, innovation)
        self.covariance = np.dot((np.eye(self.state.shape[0]) - np.dot(kalman_gain, H)), self.covariance)


    def drone_update(self, x, y, u):
        dt = 1  # Time step between measurements
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        
        B = np.array([[0.5*(dt**2), 0],
              [0, 0.5*(dt**2)],
              [dt, 0],
              [0, dt]])


        # Define measurement matrix
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        
        self.predict(A, B, u)

        z = np.array([x, y])
        self.update(z, H)

