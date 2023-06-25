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
    L = 50  # number of measurements per anchor
    
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
                # theta_measurements = np.arctan2(Y_measurements, X_measurements)
                # mask = (X_measurements**2 + Y_measurements**2 <= R**2) & ((theta_measurements <= section_start) | (theta_measurements >= section_end))
                # dists_err *= mask
                # s = np.sum(mask)
                # if np.sum(mask) > 0 and np.sum(mask) < 1000:
                #     print('hola')
                total_dist_error_sq += np.sum((dists_err**2))#/sigma(dists_n)**2)
                total_N += np.sum(L)
        
        if total_dist_error_sq:
            rms_grid[i] = np.sqrt(total_dist_error_sq / total_N)
        else:
            rms_grid[i] = np.NaN

    return rms_grid

# loss function to optimize x1 and x2
def loss_func(x0, *args):
    x, theta1, theta2 = x0
    R, grid = args
    
    # translate positions to cartesian coordinates
    anchors = [(R*np.cos(theta), R*np.sin(theta)) for theta in [theta1, theta2]]
    anchors.append((x,0))

    rms = get_rms(grid[:, 0], grid[:, 1], anchors)
    
    return np.quantile(rms[~np.isnan(rms)], 0.95)
    # return (np.sum(rms**2))/len(rms)

def tune_hyperparams(x0, args):
    # x, theta1, theta2 = x0
    
    bnds, loss_args = args
    for i, (lb, rb) in enumerate(bnds):
        rms_i = {}
        for p_val in np.linspace(lb, rb, 50):
            params = x0
            params[i] = p_val
            rms_i[p_val] = loss_func(params, *loss_args)
        
        min_val = min(rms_i, key=rms_i.get)
        x0[i] = min_val
        
    return x0

R = 12000
wall = generate_circle_points(12000, 3000)
grid = generate_circle_grid(12000, 700)

# x0 = [-6000, np.pi*(1/4), np.pi*2*(7/8)]
# bnds = [(-R, 0), (np.pi*(1/4), np.pi/2), (np.pi*2*(7/8), np.pi*2*(9/8))]
# res_x = tune_hyperparams(x0, args=(bnds, (R, grid)))

# x_val, theta1_opt, theta2_opt = res_x
# anchors = [(R*np.cos(theta), R*np.sin(theta)) for theta in [theta1_opt, theta2_opt]]
# anchors.append((x_val,0))
# anchors = np.array(anchors)
# print(anchors)


# minimize the loss function
# result = minimize(loss_func, x0, bounds=bnds, args=(R, grid), method='L-BFGS-B')
# x_val, theta1_opt, theta2_opt = result.x

# # convert optimal anchor positions from angles to cartesian
# anchors = [(R*np.cos(theta), R*np.sin(theta)) for theta in [theta1_opt, theta2_opt]]
# anchors.append((x_val,0))
# anchors = np.array(anchors)
# print(anchors)

# define angles of x1 and x2
# thetas_guess = [np.pi*1/8, np.pi*1/9]
thetas_guess = [np.pi*6/16, 0]# -np.pi*1/64]

anchors = [(R*np.cos(theta), R*np.sin(theta)) for theta in thetas_guess]
anchors.append((-2500,0))
anchors = np.array(anchors)


# Generate points along a circle with a cut-out piece of 45 degree

rms = get_rms(grid[:, 0], grid[:, 1], anchors)
rms_95 = np.quantile(rms[~np.isnan(rms)], 0.95)
# rmse_normalized = (np.abs(rmses)/np.max(rmses))

# Plot results
wall += R
grid += R
anchors += R

cmap = plt.cm.get_cmap('RdYlGn')
# Get the 'winter' colormap
winter = plt.cm.get_cmap('winter')

# # Define the number of points in the colormap
n_points = 256

# # Create a skewed list of points from 0 to 1 to give more weight to green
points = np.linspace(0, 1, n_points)**0.7

# # Create a new colormap using these points
colors = cmap(points)
import matplotlib.colors as mcolors
rg_skewed = mcolors.LinearSegmentedColormap.from_list('rg_skewed', colors)

print("wall pts: %s" % str(wall.shape))
print("area pts: %s" % str(grid.shape))

plt.gca().set_aspect('equal', adjustable='box')
plt.title('95 %% Quantile of RMS: %.2f' % rms_95)


# plt.plot(grid[:,0], grid[:, 1], c = cmap(rmse_normalized),  markersize=5)
plt.plot(wall[:,0], wall[:, 1], ".", color='#222', markersize=1)

plt.scatter(grid[:,0], grid[:, 1], c=rms, cmap=rg_skewed.reversed())
plt.scatter(grid[np.isnan(rms),0], grid[np.isnan(rms), 1], color='#dfdfdf')
plt.plot(anchors[:, 0], anchors[:, 1], ".", color="red", markersize=15, zorder=2.5)
plt.show()

fig, ax = plt.subplots(1, 1)#2, figsize=(10, 5))

count, bins, ignored = ax.hist(rms, bins=30, density=True)
mu = np.mean(rms)
sd = np.std(rms)
ax.plot(bins, 1/(sd * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sd**2) ),
         linewidth=2, color='r')
# ax[1].boxplot(rms)
plt.show()