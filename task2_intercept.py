import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares,minimize
from util import generate_circle_grid, generate_circle_points
import matplotlib.colors as mcolors

# error functions
s1 = lambda r: 0.2 + 0.05*r
s2 = lambda r: 1.0 + 0.01*r
s3 = lambda r: 0.5 + 0.1*r
# s1 = lambda r: 0.2 + 0.005*r
# s2 = lambda r: 1.0 + 0.0025*r
# s3 = lambda r: 0.5 + 0.01*r
s1 = lambda r: 0.2 + 0.001*r
s2 = lambda r: 1.0 + 0.0005*r
s3 = lambda r: 0.5 + 0.002*r
sigmas = [s1, s2, s3]

def simulate_measurements(x_a, y_a, x, y, sigma, L):
    # compute the actual distance from the anchor to the object
    d = np.sqrt((x - x_a)**2 + (y - y_a)**2)
    
    # generate L simulated measurements
    return np.random.normal(d, sigma(d), L)

def squared_loss(params, *args):
    x, y = params

    anchors = np.array(args[0])
    meas_dist = np.array(args[1])
    sigmas = np.array(args[2])

    estim_dist = np.array([np.sqrt((x - anchors[i, 0])**2 + (y - anchors[i, 1])**2) for i in range(anchors.shape[0])])

    # diff = np.array([(estim_dist[i] - meas_dist[i])**2 / (1 - sigmas[i](meas_dist)**2) for i in range(anchors.shape[0])])
    diff = np.array([(estim_dist[i] - meas_dist[i])**2  for i in range(anchors.shape[0])])
    return np.sum(diff)


    # dists_real = np.sum((anchors - np.array([x, y]))**2, axis=1)
    # diffs_sq = dists_real - dists_sq    
    # return np.sum([(diffs_sq[i] / sigma(dists_real[i]))**2 for i, sigma in enumerate(sigmas)])
    # return np.sum(list(map(lambda x: 1/sigmas((d / sigmas(dists_real))**2)

def find_intersections(anchor_visible, sigmas_visible, anchor_data, initial_guess):
    intercepts = []
    initial_guess = [0, 0]  # Initial guess for the center coordinates
    anchor_data = np.array(anchor_data)

    for i in range(anchor_data.shape[1]):
        result = minimize(squared_loss, initial_guess, method='L-BFGS-B', args=(anchor_visible, anchor_data[:, i], sigmas))
        center_x, center_y = result.x
        intercepts.append([center_x, center_y])
        
    return intercepts

# loss function to optimize x1 and x2
def get_rms(X, Y, anchors, L=500):
    R = 12000
    rms_grid = np.zeros(len(X))
    measured_pnts = []
    # L = 2000  # number of measurements per anchor
    
    for i in range(len(rms_grid)):
        
        # determine angle of grid point
        g_angle = np.arctan2(Y[i], X[i])
        g_angle = np.where(g_angle < 0, g_angle + 2*np.pi, g_angle)
        
        anchor_visible = []
        sigmas_visible = []
        anchor_data = []
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
                
                # make L measurements for distance of anchor -> grid point 
                dists_n = simulate_measurements(x_i, y_i, X[i], Y[i], sigma, L)
                
                # log anchor and applied sigma with recorded data 
                anchor_visible.append([x_i, y_i])
                sigmas_visible.append(sigma)
                anchor_data.append(dists_n)

        if(len(anchor_visible) > 1):
            z = np.array([X[i], Y[i]])
            quadrant = np.ones(2)
            if z[0] < 0:
                quadrant[0] = -1
            if z[1] < 0:
                quadrant[1] = -1
            
            z_approx = find_intersections(anchor_visible, sigmas_visible, anchor_data, z)
            rms_grid[i] = np.sqrt(np.mean((z - z_approx)**2))
            measured_pnts.append(z_approx)
        else:
            rms_grid[i] = np.NaN

    return rms_grid, measured_pnts

# loss function to optimize x1 and x2
def loss_func(x0, *args):
    theta1, theta2 = x0
    R, grid = args
    
    # translate positions to cartesian coordinates
    anchors = [(R*np.cos(theta), R*np.sin(theta)) for theta in [theta1, theta2]]
    anchors.append((0,0))

    rms, datapoints = get_rms(grid[:, 0], grid[:, 1], anchors, L=10)
    
    # return np.quantile(rms, 0.95)
    
    return np.mean(rms[~np.isnan(rms)])
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
grid = generate_circle_grid(12000, 1000)
# grid = np.array([[-2000, -3000],[6000, 2000], [10000, 3000], [2000,-11000], [-7000, 5000]])


thetas_guess = [np.pi*0.25, 0]# -np.pi*1/64]
anchors = []
anchors = [(R*np.cos(theta), R*np.sin(theta)) for theta in thetas_guess]
anchors.append((0,0))
anchors = np.array(anchors)

# minimize the loss function
# x0 = thetas_guess
# # bnds = [(-R, 0), (np.pi*(1/4), np.pi/2), (np.pi*2*(7/8), np.pi*2*(9/8))]
# result = minimize(loss_func, x0, args=(R, grid), method='L-BFGS-B')
# theta1_opt, theta2_opt = result.x
# anchors = []
# anchors = [(R*np.cos(theta), R*np.sin(theta)) for theta in result.x]
# anchors.append((0,0))
# anchors = np.array(anchors)


rms, clouds = get_rms(grid[:, 0], grid[:, 1], anchors, L=10)
if len(rms[~np.isnan(rms)]):
    rms_95 = np.quantile(rms[~np.isnan(rms)], 0.95)
else:
    rms_95 = 0 # TODO: display something esle

# Plot results
wall += R
grid += R
anchors += R
clouds = np.array(clouds) + R

# Define the number of points in the colormap
vir = plt.cm.get_cmap('viridis')
n_points = 256
points = np.linspace(0, 1, n_points)**0.9
vir_skewed = mcolors.LinearSegmentedColormap.from_list('vir_s', vir(points))


print("wall pts: %s" % str(wall.shape))
print("area pts: %s" % str(grid.shape))
print("95 %% quantile: %.2f" % rms_95)

fig, ax = plt.subplots()

plt.gca().set_aspect('equal', adjustable='box')
plt.title('95 %% Quantile of RMS: %.2f' % rms_95)


# plt.plot(grid[:,0], grid[:, 1], c = cmap(rmse_normalized),  markersize=5)
plt.plot(wall[:,0], wall[:, 1], ".", color='#222', markersize=1)


plt.scatter(grid[np.isnan(rms),0], grid[np.isnan(rms), 1], color='#dfdfdf')
plt.scatter(grid[:,0], grid[:, 1], c=rms[:], vmin=0, cmap=vir.reversed())

for i in range(clouds.shape[0]):
    plt.plot(clouds[i, :,0], clouds[i, :, 1], "r.", markersize=0.5)
plt.plot(anchors[:, 0], anchors[:, 1], ".", color="red", markersize=15, zorder=2.5)
plt.colorbar(label='RMS Error')
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.xlabel("[mm]")
plt.ylabel("[mm]")
plt.show()

fig, ax = plt.subplots(1, 1)#2, figsize=(10, 5))

count, bins, ignored = ax.hist(rms, bins=50, density=True)
mu = np.mean(rms)
sd = np.std(rms)
ax.plot(bins, 1/(sd * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sd**2) ),
         linewidth=2, color='r')

plt.axvline(x=rms_95, color='r')
# Get current ticks
# ticks = plt.xticks()[0]

# # Add a tick at the x-value
# ticks = np.append(ticks, rms_95)
# plt.xticks(ticks)

# Add a label at the x-value (adjust the y-value to position the label)
plt.text(rms_95, 0.05, '95 % Quantil', rotation=90)

# ax.plot([rms_95, rms_95], [0, mu], 'r')
# ax.boxplot(rms)

plt.show()

# task 3 besser als task 2?

# histogram wegen unterschied was optimierer sieht und tatsächlich rauskommt