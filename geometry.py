import numpy as np
import matplotlib.pyplot as plt

def generate_circle_grid(radius, step_size):
    # Define the bounds of the grid based on the radius of the circle
    x = np.arange(-radius, radius + step_size, step_size)
    y = np.arange(-radius, radius + step_size, step_size)

    # Create a grid of points
    x_coords, y_coords = np.meshgrid(x, y)

    # Initialize an empty list to hold the points inside the circle
    circle_points = []

    angle_max = np.pi*(5/4)
    angle_min = np.pi

    for i in range(x_coords.shape[0]):
        for j in range(y_coords.shape[1]):
            
            # check if points lie within circle and desired range of angles
            if x_coords[i, j]**2 + y_coords[i, j]**2 <= radius**2:
                # Convert the point to polar coordinates
                r = np.sqrt(x_coords[i, j]**2 + y_coords[i, j]**2)
                theta = np.arctan2(y_coords[i, j], x_coords[i, j])
                theta = theta + 2 * np.pi if theta < 0 else theta  # Make sure theta is in range [0, 2pi]          
                if theta < angle_min or theta > angle_max:
                    circle_points.append((x_coords[i, j], y_coords[i, j]))

    return np.array(circle_points) + radius

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
    
    # Add radius to make all points > 0
    return np.array(points) + radius

# Generate points along a circle with a cut-out piece of 45 degree
wall = generate_circle_points(1200, 300)
grid = generate_circle_grid(1200, 80)

print("wall pts: %s" % str(wall.shape))
print("area pts: %s" % str(grid.shape))

# Plot results
plt.gca().set_aspect('equal', adjustable='box')  # To keep the circle round

plt.plot(wall[:,0], wall[:, 1], ".b",  markersize=5)
plt.plot(grid[:,0], grid[:, 1], "+g",  markersize=5)
plt.show()