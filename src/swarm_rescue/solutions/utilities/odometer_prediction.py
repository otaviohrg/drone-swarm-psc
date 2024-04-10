import numpy as np
from scipy.interpolate import UnivariateSpline

class Odometer_prediction:
    def __init__(self):
        self.initial_x = None
        self.initial_y = None
        self.dx = np.array([0,0,0,0]) 
        self.dy = np.array([0,0,0,0])


    def update (self, dist, alpha, theta) :
        dx = dist*np.cos(alpha)
        dy = dist*np.sin(alpha)
        self.dx = np.append(self.dx, dx)
        self.dy = np.append(self.dy, dy)

        #if(len(self.dx) > 30):
        #    self.initial_x += self.integ_bruit_dx()
        #    self.initial_y += self.integ_bruit_dy()
        #    self.dx = np.array([0,0,0,0]) 
        #    self.dy = np.array([0,0,0,0])
    

    def lissage_dx(self):
        # séquence de temps basée sur la longueur de self.dx
        Time = np.linspace(1, len(self.dx), len(self.dx))
        # spline de lissage à self.dx en fonction du temps
        spline = UnivariateSpline(Time, self.dx, s=.1)
        return spline
    

    def integ_bruit_dx(self) :
        smooth_dx = self.lissage_dx()
        # l'intégrale de la spline sur toute la plage temporelle
        a = 1  # Début de l'intervalle de temps
        b = len(self.dx)  # Fin de l'intervalle de temps
        integral = smooth_dx.integral(a, b)
        return integral
    

    def lissage_dy(self):
        # séquence de temps basée sur la longueur de self.dx
        Time = np.linspace(1, len(self.dy), len(self.dy))
        # spline de lissage à self.dx en fonction du temps
        spline = UnivariateSpline(Time, self.dy, s=.1)
        return spline
    

    def integ_bruit_dy(self) :
        smooth_dy = self.lissage_dy()
        # l'intégrale de la spline sur toute la plage temporelle
        a = 1  # Début de l'intervalle de temps
        b = len(self.dy)  # Fin de l'intervalle de temps
        integral = smooth_dy.integral(a, b)
        return integral
        