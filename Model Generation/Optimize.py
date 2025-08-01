import numpy as np
import numba
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def fibonacci_sphere(samples=1, randomize=True):
    rnd = 1.
    if randomize:
        rnd = np.random.random() * samples

    points = []
    offset = 2./samples
    increment = np.pi * (3. - np.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y**2)

        phi = ((i + rnd) % samples) * increment

        x = np.cos(phi) * r
        z = np.sin(phi) * r

        points.append([x, y, z])

    return np.array(points)

@numba.njit
def spherical_to_cartesian(spherical_coords):
    r, theta, phi = spherical_coords[:, 0], spherical_coords[:, 1], spherical_coords[:, 2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack((x, y, z), axis=-1)

@numba.njit
def cartesian_to_spherical(cartesian_coords):
    x, y, z = cartesian_coords[:, 0], cartesian_coords[:, 1], cartesian_coords[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return np.stack((r, theta, phi), axis=-1)

def generate_initial_points(N, r_inner, r_outer):
    # Generate points uniformly on the unit sphere
    points_on_sphere = fibonacci_sphere(samples=N)

    # Convert to spherical coordinates
    spherical_coords = cartesian_to_spherical(points_on_sphere)

    # Scale the radial component to fit within the spherical shell
    spherical_coords[:, 0] = np.random.uniform(r_inner, r_outer, size=N)

    # Convert back to Cartesian coordinates for plotting
    cartesian_coords = spherical_to_cartesian(spherical_coords)

    return cartesian_coords


def calculateEuclideanDistancesMatrix(x, y, epsCutoff=8):
    x_square = np.sum(x**2, axis=1).reshape(-1,1)
    if x is y:
        y_square = x_square.T
    else:
        y_square = np.sum(y**2, axis=1).reshape(1,-1)
    distances = np.dot(x, y.T)
    # use inplace operation to accelerate
    distances *= -2
    distances += x_square
    distances += y_square
    #distances[distances > epsCutoff**2] = np.inf
    return distances


@numba.njit
def potential_energy(points):
    epsilon = 1e-5
    energy = 0.0
    N = len(points)
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(points[i] - points[j])
            energy += 1.0 / (dist + epsilon)**2
        
    return energy


def optimize_points(points):
    def objective(x):
        reshaped_points = x.reshape(-1, 3)
        return potential_energy(spherical_to_cartesian(reshaped_points))
    
    def print_progress(xk):
        reshaped_points = xk.reshape(-1, 3)
        current_energy = potential_energy(spherical_to_cartesian(reshaped_points))
        print(f"Current potential energy: {current_energy}")

    initial_spherical_points = cartesian_to_spherical(points)
    
    bounds = [(r_inner, r_outer), (0, np.pi), (- np.pi,  np.pi)] * len(initial_spherical_points)
    
    result = minimize(objective, initial_spherical_points.flatten(), method='L-BFGS-B', 
                      bounds=bounds, callback=print_progress, options={'disp': True, 'iprint':99, 
                                                                       'maxfun':10000})
    optimized_spherical_points = result.x.reshape(-1, 3)
    optimized_cartesian_points = spherical_to_cartesian(optimized_spherical_points)
    return optimized_cartesian_points

def save_to_xyz(points, filename):
    N = len(points)
    with open(filename, 'w') as file:
        file.write(f"{N}\n")
        file.write("Optimized points within a spherical shell\n")
        for point in points:
            file.write(f"C {point[0]} {point[1]} {point[2]}\n")

# Parameters
N = 337  # Number of points
r_inner = 4
r_outer = 9

# Generate initial points
initial_points = generate_initial_points(N, r_inner, r_outer)

# Optimize points
optimized_points = optimize_points(initial_points)

for i in range(4):
    optimized_points = optimize_points(optimized_points)
    save_to_xyz(optimized_points, 'optimized_points2.xyz')

# Plot the optimized points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(optimized_points[:, 0], optimized_points[:, 1], optimized_points[:, 2])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Save the optimized points to XYZ format
save_to_xyz(optimized_points, 'optimized_points2.xyz')