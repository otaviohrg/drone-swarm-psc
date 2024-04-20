import numpy as np
import os
import tensorflow as tf
from scipy.spatial import distance

class MyDroneLocalization:
    ''' Strategy for no-GPS zones.
        Using data collected using the gps, we trained models to predict relative displacements
    '''
    def __init__(self):
        ''' we will use a tflite model that's already been trained to make predictions '''
        # Construct models' file path
        script_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_directory, 'gps_model.tflite')

        #Procedure to generate predictions from TFLite model
        self.lite_model = tf.lite.Interpreter(file_path)
        self.lite_model.allocate_tensors()
        self.lite_model_input_index = self.lite_model.get_input_details()[0]["index"]
        self.lite_model_output_index = self.lite_model.get_output_details()[0]["index"]

        # Used for discretization (set empirically)
        self.scaling_factor = 45
        self.historic_size = 40
        self.last_command = {}
        self.historic_gps = np.empty((0, 2), dtype=float)
        self.last_angle = None
        self.x = None
        self.y = None
        
    def predict_position(self, command, angle, x, y):
        # Convert command to tensorflow tensor
        x_tensor = tf.convert_to_tensor([[command['forward'], command['lateral'], command['rotation'], angle]], dtype=np.float32)

        # Get prediction from model
        self.lite_model.set_tensor(self.lite_model_input_index, x_tensor)
        self.lite_model.invoke()
        prediction =  self.lite_model.get_tensor(self.lite_model_output_index)[0]

        predicted_displacement_x = prediction[0]
        predicted_displacement_y = prediction[1]
        # Add the predicted x-coordinate
        predicted_x = x + predicted_displacement_x
        # Add the predicted y-coordinate
        predicted_y = y + predicted_displacement_y
        return predicted_x, predicted_y

    def predict_angle(self, command, angle):
        #rotation = 1 => 0,2 rad per step
        dtheta = command["rotation"]*0.08969095063691242
        return angle + dtheta
    
    def get_gps_values(self, drone):
            '''Return GPS values if available, else use no gps strategy'''
            gps_x, gps_y = None, None
            if drone.gps_is_disabled():
                last_position = self.historic_gps[-1]
                predicted_x, predicted_y = self.predict_position(self.last_command, self.last_angle, last_position[0], last_position[1])

                gps_x, gps_y = predicted_x, predicted_y
            else:
                gps_x, gps_y = drone.measured_gps_position()[0], drone.measured_gps_position()[1]
            
            self.historic_gps = np.append(self.historic_gps, [[gps_x, gps_y]], axis=0)
            if len(self.historic_gps) > self.historic_size:
                self.historic_gps = np.delete(self.historic_gps, 0, 0)

            return round(gps_x/self.scaling_factor), round(gps_y/self.scaling_factor)

    def get_compass_values(self, drone):

        compass_angle = None
        if drone.compass_is_disabled():
            last_command = self.last_command

            new_angle = self.predict_angle(last_command, self.last_angle)
            compass_angle = new_angle
        else:
            compass_angle = drone.measured_compass_angle()

        self.last_angle = compass_angle
        return compass_angle
    
    def update_historic_commands(self, command):
        self.last_command = command

    def get_edge_info(self):
        if(len(self.historic_gps) > 1):
            position = self.historic_gps[-1]
            prev_position = self.historic_gps[-2]

            discrete_prev_position = (round(prev_position[0]/self.scaling_factor), round(prev_position[1]/self.scaling_factor))
            discrete_position = (round(position[0]/self.scaling_factor), round(position[1]/self.scaling_factor))

            weight = distance.euclidean(discrete_prev_position, discrete_position)
            return discrete_prev_position, discrete_position, weight
        else:
            return None, None, None