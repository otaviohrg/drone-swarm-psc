import numpy as np
import re
import os
import random
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

        #same thing for compass model
        file_path2 = os.path.join(script_directory, 'compass_model.tflite')
        self.lite_model2 = tf.lite.Interpreter(file_path2)
        self.lite_model2.allocate_tensors()
        self.lite_model_input_index2 = self.lite_model2.get_input_details()[0]["index"]
        self.lite_model_output_index2 = self.lite_model2.get_output_details()[0]["index"]
        
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

    def predict_angle(self, command, angle):
        # Convert command to tensorflow tensor
        x_tensor = tf.convert_to_tensor([[command['rotation'], angle]], dtype=np.float32)

        # Get prediction from model
        self.lite_model2.set_tensor(self.lite_model_input_index2, x_tensor)
        self.lite_model2.invoke()
        prediction =  self.lite_model2.get_tensor(self.lite_model_output_index2)[0][0]
        return prediction
