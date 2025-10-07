import numpy as np

class BergmanMinimalModel:
    """
    Bergman minimal model of glucose-insulin dynamics.
    G: plasma glucose concentration (mg/dL)
    X: remote insulin effect (1/min)
    I: plasma insulin concentration (uU/mL)
    """
    def __init__(self, Gb=100, Ib=15):
        # Physiological constants
        self.p1 = 0.03  # 1/min, glucose effectiveness
        self.p2 = 0.02  # 1/min, insulin disappearance rate
        self.p3 = 1.5e-5  # 1/min^2 per uU/mL, insulin sensitivity

        # Basal glucose and insulin
        self.G = Gb
        self.X = 0.0
        self.I = Ib

    def step(self, dt, insulin_input=0.0, meal_glucose=0.0):
        """
        Perform a single time-step update.

        :param dt: timestep (minutes)
        :param insulin_input: exogenous insulin (uU/mL per min)
        :param meal_glucose: exogenous glucose (mg/dL per min)
        :return: updated glucose value
        """
        dG = -self.p1 * (self.G - 100) - self.X * self.G + meal_glucose
        dX = -self.p2 * self.X + self.p3 * (self.I - 15)

        self.G += dt * dG
        self.X += dt * dX
        self.I += dt * (insulin_input - (self.I - 15) / 30)  # Insulin decay to baseline

        return self.G

    def reset(self, G=100, I=15):
        self.G = G
        self.X = 0.0
        self.I = I
