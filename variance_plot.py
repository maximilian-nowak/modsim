import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares,minimize
from util import generate_circle_grid, generate_circle_points, get_wall_patches
import matplotlib.colors as mcolors

# error functions
# s1 = lambda r: 0.2 + 0.05*r
# s2 = lambda r: 1.0 + 0.01*r
# s3 = lambda r: 0.5 + 0.1*r
# s1 = lambda r: 0.2 + 0.005*r
# s2 = lambda r: 1.0 + 0.0025*r
# s3 = lambda r: 0.5 + 0.01*r
s1 = lambda r: 0.2 + 0.001*r
s2 = lambda r: 1.0 + 0.0005*r
s3 = lambda r: 0.5 + 0.002*r
sigmas = [s1]#, s2, s3]
np.random.seed(1)
def simulate_measurements(x_a, y_a, x, y, sigma, L):
    
    # compute the actual distance from the anchor to the object
    d = np.sqrt((x - x_a)**2 + (y - y_a)**2)
    
    # generate L simulated measurements
    return np.random.normal(d, sigma(d), L)

def squared_loss2(params, *args):
    x, y = params
    anchors = np.array(args[0])
    meas_dist = np.array(args[1])
    sigmas = np.array(args[2])
    
    estim_dist = np.array([np.sqrt((x - anchors[i, 0])**2 + (y - anchors[i, 1])**2) for i in range(anchors.shape[0])])
    diff_sq = np.array([(estim_dist[i] - meas_dist[i])**2 / (1 + sigmas[i](meas_dist)**2) for i in range(anchors.shape[0])])

    return np.sum(diff_sq)

def squared_loss(params, *args):
    x, y = params

    anchors = np.array(args[0])
    meas_dist = np.array(args[1])
    sigmas = np.array(args[2])

    estim_dist = np.array([np.sqrt((x - anchors[i, 0])**2 + (y - anchors[i, 1])**2) for i in range(anchors.shape[0])])

    # diff_sq = np.array([(estim_dist[i] - meas_dist[i])**2 / (0.5+sigmas[i](meas_dist)**2) for i in range(anchors.shape[0])])
    # diff = np.array([(estim_dist[i] - meas_dist[i])**2 / (1 + sigmas[i](meas_dist)**2) for i in range(anchors.shape[0])])
    diff_sq = np.array([(estim_dist[i] - meas_dist[i])**2  for i in range(anchors.shape[0])])
    return np.sum(diff_sq)


def find_intersections(anchor_visible, sigmas_visible, anchor_data, initial_guess):
    intercepts = []
    ig = [0, 0]  # Initial guess for the center coordinates
    ig = initial_guess
    anchor_data = np.array(anchor_data)
    anchor_visible = np.array(anchor_visible)
    sigmas_visible = np.array(sigmas_visible)

    for i in range(anchor_data.shape[1]):

        r_i = anchor_data[:, i]
        # if(len(r_i) >= 3 and np.max(r_i) > R):
        #     idx_sorted = np.argsort(r_i)
        #     r_i = r_i[idx_sorted][:2]
        #     b = np.linalg.norm(anchor_visible[idx_sorted][:2] - anchor_visible[idx_sorted][:2])
        #     if((b**2 + r_i[0]**2 - r_i[1]**2) / (2*b))**2 < r_i[0]**2:  
        #         temp_val = (b**2 + r_i[0]**2 - r_i[1]**2) / (2 * b)
        #         ig = [temp_val, np.sqrt(r_i[0]**2 - temp_val**2)]
        #     else:
        #         ig = [0,0]
            
        # rs = anchor_data[:, i]
        # if(len(rs) >= 3 and np.max(rs) > R):
        #     idx_sorted = np.argsort(rs)
        #     result = minimize(squared_loss2, ig, method='L-BFGS-B', args=(anchor_visible[idx_sorted][-2:], rs[idx_sorted][-2:], sigmas_visible[idx_sorted][-2:]))
            
        result = minimize(squared_loss2, ig, method='L-BFGS-B', args=(anchor_visible, anchor_data[:, i], sigmas_visible))
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
            z_approx = find_intersections(anchor_visible, sigmas_visible, anchor_data, z)
            rms_grid[i] = np.sqrt(np.mean((z - z_approx)**2))
            measured_pnts.append(z_approx)
        else:
            rms_grid[i] = np.NaN

    return rms_grid, measured_pnts

def get_variances(X, Y, anchor, sigma):
    variances = np.zeros(len(X))
    for i in range(len(variances)):
        
        # determine distance to current point
        d = np.sqrt((X[i]-anchor[0])**2 + (Y[i]-anchor[1])**2)
        
        # determine variance 
        variances[i] = sigma(d)**2
    
    return variances
        
R = 12000
wall = generate_circle_points(12000, 3000)
grid = generate_circle_grid(12000, 850) # 850

thetas_guess = [np.pi*1/8]#, np.pi*0.14, -np.pi*2]# -np.pi*1/64]
anchors = []
# anchors = [(R*np.cos(theta), R*np.sin(theta)) for theta in thetas_guess]
anchors.append((0,0))
anchors = np.array(anchors)



# rms, clouds = get_rms(grid[:, 0], grid[:, 1], anchors, L=10)
variances1 = get_variances(grid[:, 0], grid[:, 1], anchors[0], s1)
variances2 = get_variances(grid[:, 0], grid[:, 1], anchors[0], s2)
variances3 = get_variances(grid[:, 0], grid[:, 1], anchors[0], s3)

# Plot results
# wall += R
# grid += R
# anchors += R

# Define the number of points in the colormap
vir = plt.cm.get_cmap('viridis')
n_points = 256
points = np.linspace(0, 1, n_points)**0.9
vir_skewed = mcolors.LinearSegmentedColormap.from_list('vir_s', vir(points))


# print("wall pts: %s" % str(wall.shape))
print("area pts: %s" % str(grid.shape))

# fig, ax = plt.subplots()
# plt.plot(np.linspace(0, 24000, len(variances1)), variances1, ".")
# plt.plot(np.linspace(0, 24000, len(variances2)), variances2, ".")
# plt.plot(np.linspace(0, 24000, len(variances3)), variances3, ".")
# plt.show()

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(12, 4))  # Adjust figsize as needed
fig.suptitle('Variance of the 3 measurement devices across the grid')
gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])  # Divide the figure into 4 columns, with the last column for the colorbar

axes = [fig.add_subplot(gs[0, i]) for i in range(3)]  # Create subplots in the first 3 columns
cbar_ax = fig.add_subplot(gs[0, 3])  # Create a subplot for the colorbar in the last column

for i, data in enumerate([variances1, variances2, variances3]):
    ax = axes[i]

    ax.set_aspect('equal', adjustable='box')
    ax.set_title('device %s' % str(i+1))

    for wall_patch in get_wall_patches(R):
        ax.add_patch(wall_patch)

    scatter = ax.scatter(grid[:, 0], grid[:, 1], c=data, s=60, vmin=0, vmax=500, cmap='viridis_r')

    ax.plot(anchors[:, 0], anchors[:, 1], ".", color="red", markersize=15, zorder=2.5)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel(r"distance [mm] $(\times 10^{3}$)")
    ax.set_ylabel(r"distance [mm] $(\times 10^{3}$)")
    
    # Adjust the y-axis and x-axis tick labels using a FuncFormatter
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}'))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}'))

# Create a colorbar using the ScalarMappable object
cbar = plt.colorbar(scatter, cax=cbar_ax, label='Variance')

plt.tight_layout()  # Optional: adjust spacing between subplots
plt.show()


# fig, ax = plt.subplots(1, 1)#2, figsize=(10, 5))

# count, bins, ignored = ax.hist(rms, bins=50, density=True)
# mu = np.mean(rms)
# sd = np.std(rms)
# ax.plot(bins, 1/(sd * np.sqrt(2 * np.pi)) *
#                np.exp( - (bins - mu)**2 / (2 * sd**2) ),
#          linewidth=2, color='r')

# plt.axvline(x=rms_95, color='r')
# # Get current ticks
# # ticks = plt.xticks()[0]

# # # Add a tick at the x-value
# # ticks = np.append(ticks, rms_95)
# # plt.xticks(ticks)

# # Add a label at the x-value (adjust the y-value to position the label)
# plt.text(rms_95, 0.05, '95 % qantile', rotation=90)
# plt.title('RMS distribution of grid points')
# plt.xlabel('RMS value')
# plt.ylabel('density')
# # ax.plot([rms_95, rms_95], [0, mu], 'r')
# # ax.boxplot(rms)

# plt.show()

# task 3 besser als task 2?

# histogram wegen unterschied was optimierer sieht und tatsächlich rauskommt