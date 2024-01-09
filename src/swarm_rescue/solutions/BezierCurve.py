import numpy as np
import matplotlib.pyplot as plt

class BezierCurve:
    def __init__(self, control_points):
        self.control_points = np.array(control_points)

    def binomial_coefficient(self, n, k):
        """Calculate binomial coefficient (n choose k)."""
        return np.math.factorial(n) // (np.math.factorial(k) * np.math.factorial(n - k))

    def bernstein_basis(self, i, n, t):
        """Calculate the Bernstein basis polynomial."""
        return self.binomial_coefficient(n, i) * (t ** i) * ((1 - t) ** (n - i))

    def bezier_curve(self, t):
        """Calculate the Bezier curve at parameter t."""
        n = len(self.control_points) - 1
        curve_point = np.zeros_like(self.control_points[0], dtype=float)

        for i, control_point in enumerate(self.control_points):
            curve_point += self.bernstein_basis(i, n, t) * control_point

        return curve_point

    def generate_curve_points(self, num_points=100):
        """Generate points along the Bezier curve."""
        t_values = np.linspace(0, 1, num_points)
        curve_points = np.array([self.bezier_curve(t) for t in t_values])
        return curve_points
    

    def output_curve_image(self):
        # Generate points along the curve
        curve_points = self.generate_curve_points()
        # Plot the Bezier curve
        plt.plot(self.control_points[:, 0], self.control_points[:, 1], 'ro-', label='Control Points')
        plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='Bezier Curve')
        plt.title('Bezier Curve')
        plt.legend()
        #Output plot to image
        plt.savefig('foo.png')