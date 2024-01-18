import numpy as np
import re
import os
from sklearn.linear_model import LinearRegression

class Localization:
    ''' Strategy for no-GPS zones
        Using data previously collected using the gps,
        we'll train a regression model to predict relative displacements
    '''
    def __init__(self):
        ''' Train model '''
        # Construct training data file path
        script_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_directory, 'training_data_localization.txt')

        train_X = np.empty((0, 4))
        train_Y = np.empty((0, 2))

        with open(file_path, 'r') as file:
            for line in file:
                values = self.extract_values(line)
                if values is not None:
                    forward, lateral, rotation, grasper, x, y = values[0], values[1], values[2], values[3], values[4], values[5]
                    train_X = np.append(train_X, [[forward, lateral, rotation, grasper]], axis=0)
                    train_Y = np.append(train_Y, [[x, y]], axis=0)
        self.reg = LinearRegression().fit(train_X, train_Y)

    def extract_values(self, line):
        # Regular expression to process training data
        pattern = r"{'forward': ([-0-9.]+), 'lateral': ([-0-9.]+), 'rotation': ([-0-9.]+), 'grasper': ([-0-9.]+)} ([-0-9.]+) ([-0-9.]+)"

        match = re.match(pattern, line)
        if match:
            forward, lateral, rotation, grasper, x, y = map(float, match.groups())
            return forward, lateral, rotation, grasper, x, y
        else:
            # Handle the case where the line doesn't match the expected pattern
            return None  
        
    def predict_position(self, current_position, command):
        # Convert command to numpy array
        command_array = np.array([command['forward'], command['lateral'], command['rotation'], command['grasper']])
        # Predict using the trained regression model
        predicted_displacement = self.reg.predict(command_array.reshape(1, -1))
        predicted_x = current_position[0] + predicted_displacement[0,0]
        predicted_y = current_position[1] + predicted_displacement[0,1]
        return predicted_x, predicted_y