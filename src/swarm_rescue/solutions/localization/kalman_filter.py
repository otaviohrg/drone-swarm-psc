import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, noisenogps, noisewtgps, drone):
        dt = 1  # Time step between measurements
        a=0.52
        r=1-a/5.75
        self.drone = drone
        self.noisenogps = noisenogps
        self.noisewtgps = noisewtgps
        self.Q=np.diag([5]*3+[1]*5)
        self.A = np.array([[1, 0, 0, dt, 0, 0, 0, 0],
                           [0, 1, 0, 0, dt, 0, 0, 0],
                           [0, 0, 1, 0, 0, dt, 0, 0],
                           [0, 0, 0, r, 0, 0, dt*a, 0],
                           [0, 0, 0, 0, r, 0, 0, dt*a],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1]])
        self.Hn =np.array([[0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1]])
        self.Hw =np.eye(8)
        #self.x=[x,y,theta,dx,dy,dtheta,d²x,d²y]
        #self.z=[dx,dy,dtheta,d²x,d²y]
        #self.R=np.diag([1,1,1,1,1])
        print("Starting with right version of KalmanFilter \n\n\n")
        
        self.state = initial_state  # Initial state estimate
        self.covariance = initial_covariance  # Initial covariance estimate
        #self.process_noise = process_noise  # Process noise covariance matrix
        #self.measurement_noise = measurement_noise  # Measurement noise covariance matrix

    #def predict(self, A, B, u):
    #    # Predict state and covariance
    #    self.state = np.dot(A, self.state) + np.dot(B, u)
    #    self.covariance = np.dot(np.dot(A, self.covariance), A.T) + self.process_noise

    def updatenogps(self, z):
        H=self.Hn
        A=self.A
        Q=self.Q
        # Update state and covariance based on measurement
        self.state=A@self.state
        self.covariance=A@self.covariance@A.T + Q

        innovation_covariance = np.dot(np.dot(H, self.covariance), H.T) + self.noisenogps
        kalman_gain = (self.covariance@ H.T)@ np.linalg.inv(innovation_covariance)

        predicted_measurement = np.dot(H, self.state)  # Calculate predicted measurement
        innovation = z - predicted_measurement
        
        self.state = self.state + np.dot(kalman_gain, innovation)
        
        self.covariance = self.covariance - kalman_gain@H@ self.covariance


    def updatewtgps(self, z):
        #self.x=[x,y,theta,dx,dy,dtheta,d²x,d²y]
        H=self.Hw
        A=self.A
        Q=self.Q
        # Update state and covariance based on measurement
        self.state=A@self.state
        self.covariance=A@self.covariance@A.T + Q

        innovation_covariance = np.dot(np.dot(H, self.covariance), H.T) + self.noisewtgps
        kalman_gain = (self.covariance@ H.T)@ np.linalg.inv(innovation_covariance)

        predicted_measurement = np.dot(H, self.state)  # Calculate predicted measurement
        innovation = z - predicted_measurement
        
        self.state = self.state + np.dot(kalman_gain, innovation)
        
        self.covariance = self.covariance - kalman_gain@H@ self.covariance

    def drone_update(self, command, gpson):
        if gpson:
            #self.z=[x,y,theta,dx,dy,dtheta,d²x,d²y]
            cos,sin=np.cos,np.sin
            #print("Command :",command)
            #print("Odometer :",self.drone.odometer_values())
            (x,y)=self.drone.gps_values()
            theta=self.drone.compass_values()
            (dist,alpha,dtheta)=self.drone.odometer_values()
            #alpha =0
            #theta=0
            angle = self.state[2]+alpha#+self.state[5]/2
            z=[x,y,theta,dist*cos(angle),dist*sin(angle),dtheta,
            command["forward"]*cos(angle)-command["lateral"]*sin(angle),
            command["forward"]*sin(angle)+command["lateral"]*cos(angle)]
            print(len(z))
            self.updatewtgps(z)
        else:
            #self.z=[dx,dy,dtheta,d²x,d²y]
            cos,sin=np.cos,np.sin
            #print("Command :",command)
            #print("Odometer :",self.drone.odometer_values())
            (dist,alpha,dtheta)=self.drone.odometer_values()
            #alpha =0
            #theta=0
            angle = self.state[2]+alpha#+self.state[5]/2
            z=[dist*cos(angle),dist*sin(angle),dtheta,
            command["forward"]*cos(angle)-command["lateral"]*sin(angle),
            command["forward"]*sin(angle)+command["lateral"]*cos(angle)]
            
            self.updatenogps(z)