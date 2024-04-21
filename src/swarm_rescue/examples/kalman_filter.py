import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, measurement_noise, drone):
        dt = 1  # Time step between measurements
        r = 100
        a = 20
        self.drone = drone
        self.measurement_noise = measurement_noise
        self.A = np.array([[1, 0, 0, dt, 0, 0, 0, 0],
                           [0, 1, 0, 0, dt, 0, 0, 0],
                           [0, 0, 1, 0, 0, dt, 0, 0],
                           [0, 0, 0, r, 0, 0, dt*a, 0],
                           [0, 0, 0, 0, r, 0, 0, dt*a],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0]])
        self.H =np.array([[0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1]])
        #self.x=[x,y,theta,dx,dy,dtheta,d²x,d²y]
        #self.z=[dx,dy,dtheta,d²x,d²y]
        self.R=np.diag([1,1,1,1,1])
        
        self.state = initial_state  # Initial state estimate
        self.covariance = initial_covariance  # Initial covariance estimate
        #self.process_noise = process_noise  # Process noise covariance matrix
        #self.measurement_noise = measurement_noise  # Measurement noise covariance matrix

    #def predict(self, A, B, u):
    #    # Predict state and covariance
    #    self.state = np.dot(A, self.state) + np.dot(B, u)
    #    self.covariance = np.dot(np.dot(A, self.covariance), A.T) + self.process_noise

    def update(self, z):
        H=self.H
        A=self.A
        # Update state and covariance based on measurement
        self.state=A@self.state
        self.covariance=A@self.covariance@A.T #+ Q

        innovation_covariance = np.dot(np.dot(H, self.covariance), H.T) + self.measurement_noise
        kalman_gain = (self.covariance@ H.T)@ np.linalg.inv(innovation_covariance)

        predicted_measurement = np.dot(H, self.state)  # Calculate predicted measurement
        innovation = z - predicted_measurement

        self.state = self.state + np.dot(kalman_gain, innovation)
        self.covariance = self.covariance - kalman_gain@H@ self.covariance


    def drone_update(self, command):
        #self.z=[dx,dy,dtheta,d²x,d²y]
        cos,sin=np.cos,np.sin
        (dist,alpha,theta)=self.drone.odometer_values()
        angle = self.state[2]+alpha
        z=[dist*cos(angle),dist*sin(angle),theta,
           command["forward"]*cos(angle)+command["lateral"]*sin(angle),
           command["forward"]*sin(angle)+command["lateral"]*cos(angle)]
        
        self.update(z)