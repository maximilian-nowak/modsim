import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def ring_cost(params, x1, y1, R1, r1, x2, y2, R2, r2, x3, y3, R3, r3):
    x, y = params
    distance1 = np.sqrt((x - x1)**2 + (y - y1)**2) - (R1 + r1)
    distance2 = np.sqrt((x - x2)**2 + (y - y2)**2) - (R2 + r2)
    distance3 = np.sqrt((x - x3)**2 + (y - y3)**2) - (R3 + r3)
    return (distance1**2 + distance2**2 + distance3**2) / 3

def find_intersection_center(x1, y1, R1, r1, x2, y2, R2, r2, x3, y3, R3, r3):
    initial_guess = [0, 0]  # Initial guess for the center coordinates
    result = minimize(ring_cost, initial_guess, args=(x1, y1, R1, r1, x2, y2, R2, r2, x3, y3, R3, r3))
    center_x, center_y = result.x
    return center_x, center_y

def plot_2d_torus(R, err, num_points):
    r = err
    theta = np.linspace(0, 2*np.pi, num_points)

    x11 = 0 + (R+r) * np.cos(theta)
    x12 = 0 + (R-r) * np.cos(theta)
    y11 = 0 + (R+r) * np.sin(theta)
    y12 = 0 + (R-r) * np.sin(theta)

    x21 = 6 + (R+r) * np.cos(theta)
    x22 = 6 + (R-r) * np.cos(theta)
    y21 = 0 + (R+r) * np.sin(theta)
    y22 = 0 + (R-r) * np.sin(theta)
    
    x31 = 3 + (R+r) * np.cos(theta)
    x32 = 3 + (R-r) * np.cos(theta)
    y31 = 5 + (R+r) * np.sin(theta)
    y32 = 5 + (R-r) * np.sin(theta)


    c = find_intersection_center(0, 0, R, r, 6, 0, R, r, 3, 5, R, r)
    print(c)
    center_x, center_y = c
    
    plt.figure(figsize=(8, 8))
    plt.title('2D Torus')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.plot(x11, y11, 'b.')
    plt.plot(x21, y21, 'g.')
    plt.plot(x31, y31, 'm.')
    plt.plot(x12, y12, 'b.')
    plt.plot(x22, y22, 'g.')
    plt.plot(x32, y32, 'm.')
    plt.plot(center_x, center_y, 'r.')
    plt.show()


# Example usage
R = 3  # Major radius
err = 1  # Minor radius
num_points = 100  # Number of points to plot

plot_2d_torus(R, err, num_points)