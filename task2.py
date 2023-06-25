import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from util import generate_circle_grid, generate_circle_points, colorFader

# error functions
s1 = lambda r: 0.2 + 0.001*r
s2 = lambda r: 1.0 + 0.0005*r
s3 = lambda r: 0.5 + 0.002*r
sigmas = [s1, s2, s3]

def simulate_measurements(x_a, y_a, x, y, sigma, L):
    # compute the actual distance from the anchor to the object
    d = np.sqrt((x - x_a)**2 + (y - y_a)**2)
    # generate a simulated measurement
    dists_n = np.random.normal(d, sigma(d), L)
    
    # compute coordinates
    anchor_to_obj = np.array([x, y]) - np.array([x_a, y_a])
    theta_anchor_to_obj = np.arctan2(anchor_to_obj[1], anchor_to_obj[0])
    X = x_a + dists_n * np.cos(theta_anchor_to_obj)
    Y = y_a + dists_n * np.sin(theta_anchor_to_obj)

    err = dists_n - d
    return X, Y, err


# loss function to optimize x1 and x2
def get_rms(X, Y, anchors):
    R = 12000
    rms_grid = np.zeros(len(X))
    L = 1000  # number of measurements per anchor
    
    for i in range(len(rms_grid)):
        total_N = 0
        total_dist_error_sq = 0
        # determine angle of grid point
        g_angle = np.arctan2(Y[i], X[i])
        g_angle = np.where(g_angle < 0, g_angle + 2*np.pi, g_angle)
        
        for sigma, (x_i, y_i) in zip(sigmas, anchors):
            # determine angle of anchor and non-visible sector
            angle_i = np.arctan2(y_i, x_i)
            angle_i = np.where(angle_i < 0, angle_i + 2*np.pi, angle_i)
            section_start, section_end = np.pi, 5*np.pi/4
            if(angle_i > np.pi):
                section_start = angle_i - np.pi
            else:
                if(angle_i > np.pi/4):
                    section_end = angle_i + np.pi
                    
            # check if grid point is within visible region and if so, perform measurements
            if (g_angle <= section_start) or (g_angle >= section_end):
                # make L measurements 
                X_measurements, Y_measurements, dists_err = simulate_measurements(x_i, y_i, X[i], Y[i], sigma, L)
                
                # check if measurements fall within valid region
                theta_measurements = np.arctan2(Y_measurements, X_measurements)
                mask = (X_measurements**2 + Y_measurements**2 <= R**2) & ((theta_measurements <= section_start) | (theta_measurements >= section_end))
                dists_err *= mask
                
                total_dist_error_sq += (dists_err)**2
                total_N += np.sum(mask)
        
        rms_grid[i] = np.sqrt(np.sum(total_dist_error_sq) / total_N)

    return rms_grid

# loss function to optimize x1 and x2
def loss_func(thetas):
    R = 12000
    
    # translate positions to cartesian coordinates
    anchors = [(R*np.cos(theta), R*np.sin(theta)) for theta in thetas]
    anchors.append((0,0))  # add (0,0) per default
    
    # generate grid of points and compute angles
    grid = generate_circle_grid(12000, 700)
    
    return np.quantile(get_rms(grid[:, 0], grid[:, 1], anchors), 0.95)

# thetas_guess = [np.pi/2, np.pi*3/2]  # 45 and 315 degrees
# R = 12000

# # minimize the loss function
# result = minimize(loss_func, thetas_guess)

# # convert optimal anchor positions from angles to cartesian
# thetas_opt = result.x
# anchors_opt = np.array([[R*np.cos(theta), R*np.sin(theta)] for theta in thetas_opt])
# anchors_opt = np.vstack([anchors_opt, [0,0]])

# print("Optimal anchor positions (in radians):", thetas_opt)
# print("Optimal anchor positions (in cartesian coordinates): \n", anchors_opt)

# anchors = [(R*np.cos(theta), R*np.sin(theta)) for theta in thetas_opt]
# anchors.append((0,0))

R = 12000
# define angles of x1 and x2
thetas = [np.pi*3/4, np.pi*7/4]
anchors = [(R*np.cos(theta), R*np.sin(theta)) for theta in thetas]
# hardcode x3 to be at center of circle
anchors.append((0,0))
anchors_opt = np.array([[x, y] for (x, y) in anchors])

# Generate points along a circle with a cut-out piece of 45 degree
wall = generate_circle_points(12000, 3000)
grid = generate_circle_grid(12000, 700)
rmses = get_rms(grid[:, 0], grid[:, 1], anchors)
rmse_normalized = (np.abs(rmses)/np.max(rmses))

# Plot results
wall += R
grid += R
anchors_opt += R
c1 = 'red'
c2 = 'green'
n = 10
color_gradient = [colorFader(c1, c2, rn) for rn in rmse_normalized]
cmap = plt.cm.get_cmap('RdYlGn')


print("wall pts: %s" % str(wall.shape))
print("area pts: %s" % str(grid.shape))

plt.gca().set_aspect('equal', adjustable='box')
plt.title('95 %% Quantile of RMSE: %.2f' % np.quantile(rmses, 0.95))
plt.plot(wall[:,0], wall[:, 1], ".b",  markersize=3)
plt.scatter(grid[:,0], grid[:, 1], c=rmse_normalized, cmap=cmap.reversed())
# plt.plot(grid[:,0], grid[:, 1], c = cmap(rmse_normalized),  markersize=5)
plt.plot(anchors_opt[:, 0], anchors_opt[:, 1], ".r", markersize=10)
plt.show()


