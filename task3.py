import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from util import generate_circle_grid, generate_circle_points

# error functions
s1 = lambda r: 0.2 + 0.001*r
s2 = lambda r: 1.0 + 0.0005*r
s3 = lambda r: 0.5 + 0.002*r
sigmas = [s1, s2, s3]

def simulate_measurements(x_a, y_a, x, y, sigma, L):
    # compute the actual distance from the anchor to the object
    d = np.sqrt((x - x_a)**2 + (y - y_a)**2)
    
    # generate L simulated measurements
    dists_n = np.random.normal(d, sigma(d), L)
    
    # compute coordinates
    anchor_to_obj = np.array([x, y]) - np.array([x_a, y_a])
    theta_anchor_to_obj = np.arctan2(anchor_to_obj[1], anchor_to_obj[0])
    X = x_a + dists_n * np.cos(theta_anchor_to_obj)
    Y = y_a + dists_n * np.sin(theta_anchor_to_obj)

    err = dists_n - d
    return X, Y, dists_n, err


# loss function to optimize x1 and x2
def get_rms(X, Y, anchors):
    R = 12000
    rms_grid = np.zeros(len(X))
    L = 1000  # number of measurements per anchor
    
    for i in range(len(rms_grid)):
        total_N = 10**-7
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
                X_measurements, Y_measurements, dists_n, dists_err = simulate_measurements(x_i, y_i, X[i], Y[i], sigma, L)
                
                # check if measurements fall within valid region
                theta_measurements = np.arctan2(Y_measurements, X_measurements)
                mask = (X_measurements**2 + Y_measurements**2 <= R**2) & ((theta_measurements <= section_start) | (theta_measurements >= section_end))
                dists_err *= mask
                
                total_dist_error_sq += np.sum(dists_err**2)#/sigma(dists_n*mask))**2
                total_N += np.sum(mask)
        
        if total_dist_error_sq:
            rms_grid[i] = np.sqrt(total_dist_error_sq / total_N)
        else:
            rms_grid[i] = np.NaN

    return rms_grid

# loss function to optimize x1 and x2
def loss_func(thetas):
    R = 12000
    
    # translate positions to cartesian coordinates
    anchors = [(R*np.cos(theta), R*np.sin(theta)) for theta in thetas]
    anchors.append((0,0))  # add (0,0) per default
    
    # generate grid of points and compute angles
    grid = generate_circle_grid(12000, 700)

    rms = get_rms(grid[:, 0], grid[:, 1], anchors)
    
    return np.quantile(rms[~np.isnan(rms)], 0.95)


R = 12000
# thetas_guess = [np.pi*2/4, np.pi*3/2]

# # minimize the loss function
# result = minimize(loss_func, thetas_guess, method='SLSQP')

# # convert optimal anchor positions from angles to cartesian
# thetas_opt = result.x
# anchors_opt = np.array([[R*np.cos(theta), R*np.sin(theta)] for theta in thetas_opt])
# anchors_opt = np.vstack([anchors_opt, [0,0]])

# print("Optimal anchor positions (in radians):", thetas_opt)
# print("Optimal anchor positions (in cartesian coordinates): \n", anchors_opt)

# anchors = [(R*np.cos(theta), R*np.sin(theta)) for theta in thetas_opt]
# anchors.append((0,0))

# define angles of x1 and x2
anchors = []
thetas_guess = [np.pi*4/4, np.pi*3/4]

# anchors = [(R*np.cos(theta), R*np.sin(theta)) for theta in thetas_guess]
# hardcode x3 to be at center of circle

anchors.append((4000,6000))
anchors.append((8000,-6000))
anchors.append((6000,0))

anchors_opt = np.array([[x, y] for (x, y) in anchors])

# Generate points along a circle with a cut-out piece of 45 degree
wall = generate_circle_points(12000, 3000)
grid = generate_circle_grid(12000, 700)
rms = get_rms(grid[:, 0], grid[:, 1], anchors)
rms_95 = np.quantile(rms[~np.isnan(rms)], 0.95)
# rmse_normalized = (np.abs(rmses)/np.max(rmses))

# Plot results
wall += R
grid += R
anchors_opt += R

cmap = plt.cm.get_cmap('RdYlGn')

print("wall pts: %s" % str(wall.shape))
print("area pts: %s" % str(grid.shape))

plt.gca().set_aspect('equal', adjustable='box')
plt.title('95 %% Quantile of RMS: %.2f' % rms_95)


# plt.plot(grid[:,0], grid[:, 1], c = cmap(rmse_normalized),  markersize=5)
plt.plot(wall[:,0], wall[:, 1], ".", color='#222', markersize=1)

plt.scatter(grid[:,0], grid[:, 1], c=rms, cmap=cmap.reversed())
plt.scatter(grid[np.isnan(rms),0], grid[np.isnan(rms), 1], color='#dfdfdf')
plt.plot(anchors_opt[:, 0], anchors_opt[:, 1], ".", color="red", markersize=15, zorder=2.5)
plt.show()


