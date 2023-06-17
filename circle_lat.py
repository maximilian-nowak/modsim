import numpy as np
import matplotlib.pyplot as plt

def generate_line_points(start, end, n_points):
    # Generate a sequence of n_points evenly spaced between 0 and 1
    t = np.linspace(0, 1, n_points)

    # Linearly interpolate between the start and end coordinates
    x_coords = start[0] * (1 - t) + end[0] * t
    y_coords = start[1] * (1 - t) + end[1] * t

    # Zip the x and y coordinates into tuple pairs
    points = list(zip(x_coords, y_coords))

    return points

def generate_circle_points(radius, n_points):
    # calculate number of points for circle and edges relative to length of area
    full_length = (7/4)*np.pi*radius + 2*radius
    n_edge = int(((radius)/full_length) * n_points)
    n_circle = n_points - 2*n_edge
    
    # Generate n_circle points evenly spaced around circle area
    angles = np.linspace(np.pi, -3/4*np.pi, n_circle)
    circle_xs = radius * np.cos(angles)
    circle_ys = radius * np.sin(angles)
    points = list(zip(circle_xs, circle_ys))

    # Generate n_edge points evenly spaced along the edges
    edge1 = generate_line_points((circle_xs[-1], circle_ys[-1]), (0,0), n_edge+1)
    edge2 = generate_line_points((-radius, 0), (0,0), n_edge+2)
    points.extend(edge1[1:])
    points.extend(edge2[1:-1])
    
     # add radius to make all coordinates > 0
    return np.array(points) + radius

# Generate points along a circle with a cut-out piece of 45 degree
points = generate_circle_points(1200, 50)

plt.gca().set_aspect('equal', adjustable='box')  # To keep the circle round
plt.plot(points[:,0], points[:, 1], ".b",  markersize=3)
plt.show()