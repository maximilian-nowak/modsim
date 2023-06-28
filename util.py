import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def generate_circle_grid(radius, step_size):
    # Define the bounds of the grid based on the radius of the circle
    x = np.arange(-radius, radius + step_size, step_size)
    y = np.arange(-radius, radius + step_size, step_size)

    # Create a grid of points
    x_coords, y_coords = np.meshgrid(x, y)

    circle_points = []
    angle_max = np.pi*(5/4)
    angle_min = np.pi
    
    # Check for each point if it lies within the circle and within the desired angles
    for i in range(x_coords.shape[0]):
        for j in range(y_coords.shape[1]):
            
            if x_coords[i, j]**2 + y_coords[i, j]**2 <= radius**2:
                # Convert point to polar coordinates
                r = np.sqrt(x_coords[i, j]**2 + y_coords[i, j]**2)
                theta = np.arctan2(y_coords[i, j], x_coords[i, j])
                
                # Make sure theta is in range [0, 2pi]
                if theta < 0:
                    theta = theta + 2 * np.pi

                # Check if the angle lies within the desired range
                if theta < angle_min or theta > angle_max:
                    circle_points.append((x_coords[i, j], y_coords[i, j]))
    
    return np.array(circle_points)

def generate_line_points(start, end, n_points):
    # Generate points in a line
    t = np.linspace(0, 1, n_points)

    # Linearly interpolate between the start and end coordinates
    x_coords = start[0] * (1-t) + end[0] * t
    y_coords = start[1] * (1-t) + end[1] * t

    return list(zip(x_coords, y_coords))

def generate_circle_points(radius, n_points):
    # Calculate number of points for circle and edges relative to length of area
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
    
    return np.array(points)

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
colors = [(0, "blue"), (1, "green")]
cmap_name = "blue_green"
blue_green = LinearSegmentedColormap.from_list(cmap_name, colors)
cmap = plt.cm.get_cmap('winter_r')
# Get the 'winter' colormap
# # winter = plt.cm.get_cmap('winter')

# # Define the number of points in the colormap
# n_points = 256

# # Create a skewed list of points from 0 to 1 to give more weight to green
# points = np.linspace(0, 1, n_points)**0.6

# # Create a new colormap using these points
# colors = winter(points)
# winter_skewed = mcolors.LinearSegmentedColormap.from_list('winter_skewed', colors)

# cmap = plt.cm.get_cmap('terrain').reversed()