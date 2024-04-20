import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# Copy the content of the file into a string
data_string = """
1.384916395368931
1.3892168801178677
1.5844086533469808
1.5858715081103565
1.758659906147786
1.7635620502706537
1.9109299672842415
1.9150341281094487
2.082098511864814
2.0894726681088587
2.2712588550080195
2.304502832252246
2.4809349461526313
2.478040131386112
2.6515768190331954
2.6449190703719534
2.8322237459375046
2.8381809361536456
3.019986279922989
3.025309190328633
"""

# Convert the string into a numpy array
angle_data = np.array(data_string.strip().split('\n'), dtype=float)

# Generate time index
time_index = np.arange(len(angle_data)).reshape(-1, 1)

# Perform linear regression
model = LinearRegression()
model.fit(time_index, angle_data)

# Extract estimated angular velocity (slope of the linear regression line)
angular_velocity = model.coef_[0]

# Plot the original data and the linear regression line
plt.scatter(time_index, angle_data, label='Angle Data')
plt.plot(time_index, model.predict(time_index), color='red', label='Linear Regression')
plt.xlabel('Time Index')
plt.ylabel('Angle')
plt.title('Estimation of Angular Velocity')
plt.legend()
plt.show()

# Print the estimated angular velocity
print("Estimated Angular Velocity:", angular_velocity) #Estimated Angular Velocity: 0.08969095063691242
