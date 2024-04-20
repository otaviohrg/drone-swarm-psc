import numpy as np
import os
import tensorflow as tf

class MyDroneLocalization:
    ''' Strategy for no-GPS zones.
        Using data collected using the gps, we trained models to predict relative displacements
    '''
    def __init__(self):
        ''' we will use a tflite model that's already been trained to make predictions '''
        # Construct models' file path
        script_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_directory, 'gps_model.tflite')

        self.interpreter = tf.lite.Interpreter(model_path=file_path)
        self.interpreter.allocate_tensors()

        # Used for discretization (set empirically)
        self.scaling_factor = 45
        self.historic_size = 40
        self.historic_commands = np.array([])
        self.historic_gps = np.array([])
        self.historic_angle = np.array([])
        self.x = None
        self.y = None
        
    def predict_position(self, command, angle, x, y):
        # Prepare input data
        input_data = np.array([[command["forward"], command["lateral"], command["rotation"], angle, x, y]], dtype=np.float32)
        
        # Set input tensor
        input_details = self.interpreter.get_input_details()
        self.interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        self.interpreter.invoke()

        # Get output tensor
        output_details = self.interpreter.get_output_details()
        output_data = self.interpreter.get_tensor(output_details[0]['index'])

        return output_data


    def predict_angle(self, command, angle):
        #rotation = 1 => 0,2 rad per step
        dtheta = command["rotation"]*0.08969095063691242
        return angle + dtheta
    
    def get_gps_values(self, drone):
            '''Return GPS values if available, else use no gps strategy'''
            if drone.gps_is_disabled():
                last_command = self.historic_commands[-1]
                last_position = self.historic_gps[-1]
                last_angle = self.historic_angle[-1]
                predicted_x, predicted_y = self.predict_position(last_position, last_command, last_angle)

                discrete_gps_x = predicted_x/self.scaling_factor
                discrete_gps_y = predicted_y/self.scaling_factor

                return discrete_gps_x, discrete_gps_y
            else:
                return round(drone.measured_gps_position()[0]/self.scaling_factor), round(drone.measured_gps_position()[1]/self.scaling_factor)

    def get_compass_values(self, drone):
        if drone.compass_is_disabled():
            last_command = self.historic_commands[-1]
            last_angle = self.historic_angle[-1]

            new_angle = self.predict_angle(last_command, last_angle)
            return new_angle
        else:
            return drone.measured_compass_angle()
    
    def update_gps_values(self, drone):
        self.x, self.y = self.get_gps_values(drone)

        self.historic_gps.append((self.x, self.y))
        if len(self.historic_gps) > self.historic_size:
            self.historic_gps.pop(0)

    def update_compass_values(self, drone):
        self.compass_angle = self.get_compass_values(drone)

        self.historic_angle.append(self.compass_angle)
        if len(self.historic_angle) > self.historic_size:
            self.historic_angle.pop(0)