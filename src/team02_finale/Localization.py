import numpy as np
import re
import os
import tensorflow as tf
from sklearn.ensemble import GradientBoostingRegressor

class Localization:
    ''' Strategy for no-GPS zones
        Using data previously collected using the gps,
        we'll train a regression model to predict relative displacements
    '''
    def __init__(self):
        ''' we will use a tflite model that's already been trained to make predictions '''
        # Construct model's file path
        script_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_directory, 'no_gps_model.tflite')

        #Procedure to generate predictions from TFLite model
        self.lite_model = tf.lite.Interpreter(file_path)
        self.lite_model.allocate_tensors()
        self.lite_model_input_index = self.lite_model.get_input_details()[0]["index"]
        self.lite_model_output_index = self.lite_model.get_output_details()[0]["index"]

    def extract_values(self, line):
        # Regular expression to process training data
        pattern = r"{'forward': ([\d.-]+), 'lateral': ([\d.-]+), 'rotation': ([\d.-]+), 'grasper': ([\d.-]+)} ([\d.-]+) ([\d.-]+) ([\d.-]+)"

        match = re.match(pattern, line)
        if match:
            forward, lateral, rotation, grasper, angle, x, y = map(float, match.groups())
            return forward, lateral, rotation, grasper, angle, x, y
        else:
            # Handle the case where the line doesn't match the expected pattern
            return None  
        
    def predict_position(self, current_position, command, angle):
        # Convert command to tensorflow tensor
        x_tensor = tf.convert_to_tensor([[command['forward'], command['lateral'], command['rotation'], angle]], dtype=np.float32)

        # Get prediction from model
        self.lite_model.set_tensor(self.lite_model_input_index, x_tensor)
        self.lite_model.invoke()
        prediction =  self.lite_model.get_tensor(self.lite_model_output_index)[0]

        predicted_displacement_x = prediction[0]
        predicted_displacement_y = prediction[1]
        # Add the predicted x-coordinate
        predicted_x = current_position[0] + predicted_displacement_x
        # Add the predicted y-coordinate
        predicted_y = current_position[1] + predicted_displacement_y
        return predicted_x, predicted_y
