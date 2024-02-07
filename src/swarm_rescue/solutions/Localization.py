import numpy as np
import re
import os
from sklearn.ensemble import GradientBoostingRegressor

class Localization:
    ''' Strategy for no-GPS zones
        Using data previously collected using the gps,
        we'll train a regression model to predict relative displacements
    '''
    def __init__(self):
        ''' Train models '''
        # Construct training data file path
        script_directory = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_directory, 'training_data_localization.txt')

        train_X = np.empty((0, 5))
        train_Y_x = np.empty((0, 1))  # Separate target variable for x-coordinate
        train_Y_y = np.empty((0, 1))  # Separate target variable for y-coordinate

        with open(file_path, 'r') as file:
            for line in file:
                values = self.extract_values(line)
                if values is not None:
                    forward, lateral, rotation, grasper, angle, x, y = values[0], values[1], values[2], values[3], values[4], values[5], values[6]
                    train_X = np.append(train_X, [[forward, lateral, rotation, grasper, angle]], axis=0)
                    train_Y_x = np.append(train_Y_x, [[x]], axis=0)  # Add x-coordinate
                    train_Y_y = np.append(train_Y_y, [[y]], axis=0)  # Add y-coordinate

        self.reg_x = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        self.reg_y = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        self.reg_x.fit(train_X, train_Y_x)
        self.reg_y.fit(train_X, train_Y_y)

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
        # Convert command to numpy array
        x_given = np.array([command['forward'], command['lateral'], command['rotation'], command['grasper'], angle])
        # Predict using the trained regression models
        predicted_displacement_x = self.reg_x.predict(x_given.reshape(1, -1))
        predicted_displacement_y = self.reg_y.predict(x_given.reshape(1, -1))
        # Add the predicted x-coordinate
        predicted_x = current_position[0] + predicted_displacement_x[0]
        # Add the predicted y-coordinate
        predicted_y = current_position[1] + predicted_displacement_y[0]
        return predicted_x, predicted_y
