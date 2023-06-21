import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
    # Check each point in the grid to see if it lies within the circle
    for i in range(x_coords.shape[0]):
        for j in range(y_coords.shape[1]):
            # Use the equation of a circle to check if a point lies within the circle
            if x_coords[i, j]**2 + y_coords[i, j]**2 <= radius**2:
                # Convert the point to polar coordinates
                r = np.sqrt(x_coords[i, j]**2 + y_coords[i, j]**2)
                theta = np.arctan2(y_coords[i, j], x_coords[i, j])
                
                # Make sure theta is in range [0, 2pi]
                if theta < 0:
                    theta = theta + 2 * np.pi

                # Check if the angle lies within the desired range
                if theta < angle_min or theta > angle_max:
                    circle_points.append((x_coords[i, j], y_coords[i, j]))
    
    return np.array(circle_points) + radius

# define error functions
def e1(r): return 0.2 + 0.001*r
def e2(r): return 1.0 + 0.0005*r
def e3(r): return 0.5 + 0.002*r
# error_funcs = [e1, e2, e3]
error_funcs = [e1, e2, e3]

# define a penalty function that increases as the anchors get closer together
def penalty_func(anchors, R):
    penalty = 0
    for i in range(len(anchors)):
        for j in range(i+1, len(anchors)):
            # add the reciprocal of the distance between anchor i and anchor j
            penalty += (R/np.sqrt(3)) / np.sqrt((anchors[i][0] - anchors[j][0])**2 + (anchors[i][1] - anchors[j][1])**2)
    return penalty

# define the loss function
def loss_func(thetas):
    # thetas are the polar coordinates (angles) of the three anchors
    # theta1, theta2, theta3 = thetas
    R = 12000
    # anchor positions in cartesian coordinates
    anchors = [(R*np.cos(theta), R*np.sin(theta)) for theta in thetas]
    anchors.append((0,0))
    
    # for simplicity, we will compute the average loss over a grid of points in the circle
    grid = generate_circle_grid(12000, 700) - R
    X, Y = np.meshgrid(grid[:, 0], grid[:, 1])

    # compute the angles of each point (x, y) in the grid, with respect to the origin
    angles = np.arctan2(Y, X)

    # adjust the range of angles from [-pi, pi] to [0, 2*pi]
    angles = np.where(angles < 0, angles + 2*np.pi, angles)

    # mask for points outside the 1/8 circle between the angles of PI and (5/4)*PI
    mask = (X**2 + Y**2 <= R**2) & ((angles <= np.pi) | (angles >= 5*np.pi/4))

    # # compute the total error for each point in the circle
    # total_error = sum(error_func(np.sqrt((X - x_i)**2 + (Y - y_i)**2)) for error_func, (x_i, y_i) in zip(error_funcs, anchors))

    # # compute the average loss over the points in the circle
    # loss = np.sum(total_error*mask) / np.sum(mask)
    
    # compute the total squared error for each point in the circle
    total_squared_error = sum(error_func(np.sqrt((X - x_i)**2 + (Y - y_i)**2))**2 for error_func, (x_i, y_i) in zip(error_funcs, anchors))

    # compute the root mean square error (RMSE) over the points in the circle
    loss = np.sqrt(np.sum(total_squared_error*mask) / np.sum(mask))
    
    loss += penalty_func(anchors, R)
    
    # TODO: filter out 0.95 quantile ? maybe. or extra monte-carlo simulation afterwards

    return loss

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


# initial guess for anchor positions (equally spaced around the circle)
# thetas_guess = [0, 2*np.pi/3, 4*np.pi/3]
# thetas_guess = [np.pi/2, 6*np.pi/4]
thetas_guess = [0.78, 5.49]  # 45 and 315 degrees

# use Scipy's optimize.minimize function to find the anchor positions that minimize the loss function
result = minimize(loss_func, thetas_guess, method='SLSQP')

# the optimal anchor positions in polar coordinates (radians)
thetas_opt = result.x
print("Optimal anchor positions (in radians):", thetas_opt)

# convert to cartesian coordinates
R = 12000
# thetas_opt = np.where(thetas_opt > 6.28319, thetas_opt - 0.7853, thetas_opt)
# np.rad2deg(np.pi/2)
# thetas_opt = [-332.75, 27.23]
anchors_opt = np.array([[R*np.cos(theta), R*np.sin(theta)] for theta in thetas_opt]) + R
anchors_opt = np.vstack([anchors_opt, [R,R]])
print("Optimal anchor positions (in cartesian coordinates): \n", anchors_opt)


# Generate points along a circle with a cut-out piece of 45 degree
wall = generate_circle_points(12000, 3000)
grid = generate_circle_grid(12000, 700)

print("wall pts: %s" % str(wall.shape))
print("area pts: %s" % str(grid.shape))

# Plot results
plt.gca().set_aspect('equal', adjustable='box')  # To keep the circle round

plt.plot(wall[:,0], wall[:, 1], ".b",  markersize=3)
plt.plot(grid[:,0], grid[:, 1], "+g",  markersize=5)
plt.plot(anchors_opt[:, 0], anchors_opt[:, 1], ".r", markersize=10)
plt.show()
