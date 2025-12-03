import numpy as np

class RewardModifiers:
    
    def __init__(self):
        pass
    
    def cart_pole_v1_r1(self, state):
        x, theta = state[0], state[2]
        return 2**(-abs(x)/2.4) + 2**(-abs(theta)/(12 * 3.1416 / 180))
    
    def cart_pole_v1_r2(self, state):
        x, theta = state[0], state[2]
        return 1.0 - (abs(x) / 2.4)/2 - (abs(theta) / (12 * 3.1416 / 180))/2
    
    def cart_pole_v1_r3(self, state):
        theta = state[2]
        return 1 - (abs(theta) / (12 * 3.1416 / 180))**2
    
    def cart_pole_v1_r4(self, state):
        x, theta = state[0], state[2]
        
        if (abs(x) > 1.8):
            r1 = abs(x) / 2.4
        else:
            r1 = 0
            
        if (abs(theta) > (6 * 3.1416 / 180)):
            r2 = abs(theta) / (12 * 3.1416 / 180)
        else:
            r2 = 0
            
        return 1.0 - r1 - r2
    
    def acrobot_v1_r1(self, state):
        cos_theta_1 = state[0]
        return -1 - cos_theta_1
    
    def acrobot_v1_r2(self, state):
        if state[1] >=0:
            if state[3] >=0:
                return -1 - np.cos(np.arccos(state[0]) + np.arccos(state[2]))
            else:
                return -1 - np.cos(np.arccos(state[0]) - np.arccos(state[2]))
        else:
            if state[3] >=0:
                return -1 + np.cos(np.arccos(state[0]) - np.arccos(state[2]))
            else:
                return -1 + np.cos(np.arccos(state[0]) + np.arccos(state[2]))