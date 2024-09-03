import numpy as np
import matplotlib.pyplot as plt

# LBM parameters
nx, ny = 250, 250  # Grid size
lattice_length = 1
tau = 0.99  # Relaxation time
omega = 1.0 / tau  # Relaxation parameter
iterations = 1000  # Number of iterations

# D2Q9 Lattice velocities and weights
velocities = np.array([[0, 0], [1, 0], [0, 1], [-1, 0],
                      [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
weights = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)

inverse_velocity_index = [0, 3, 4, 1, 2, 7, 8, 5, 6]

# Initialize macroscopic variables
rho = np.ones((nx, ny))  # Density
u = np.zeros((nx, ny))  # Velocity in x direction
v = np.zeros((nx, ny))  # Velocity in y direction


rho[125, 125] = 0.001


def lamb_oseen_ICs():
    # inistalise with a lamb oseen-vortex at the centre
    gamma = 0.001
    centre = np.array([int(nx/2)*lattice_length, int(ny/2)*lattice_length])

    def lamb_oseen(x, y):
        r = np.sqrt(x**2 + y**2)
        coeff = gamma/(2*np.pi)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(r == 0, np.inf, r)
            u = -coeff * y / r
            v = coeff * x / r

        return [-coeff * y/r, coeff * x/r]

    # velocity
    x = np.linspace(0, nx*lattice_length, nx)
    y = np.linspace(0, ny*lattice_length, ny)

    x_rel, y_rel = [x - centre[0], y - centre[1]]
    X, Y = np.meshgrid(x_rel, y_rel)
    u, v = lamb_oseen(X, Y)
    return u, v


# Initialize distribution function (feq)
f = np.ones((nx, ny, 9)) * (rho[..., None] * weights)


def equilibrium_distribution(rho, u, v):
    usqr = u**2 + v**2
    feq = np.zeros((nx, ny, 9))

    # Calculate eu for each velocity direction
    for i in range(9):
        eu = velocities[i, 0] * u + velocities[i, 1] * \
            v  # Dot product for each direction
        feq[:, :, i] = rho * weights[i] * \
            (1 + 3 * eu + 9 / 2 * eu**2 - 3 / 2 * usqr)

    return feq

# Streaming step


def streaming(f):
    for i, vel in enumerate(velocities):
        f[:, :, i] = np.roll(f[:, :, i], shift=vel, axis=(0, 1))
    return f


# Main LBM loop
for it in range(iterations):
    feq = equilibrium_distribution(rho, u, v)
    f = omega * feq + (1 - omega) * f  # Collision step
    f = streaming(f)  # Streaming step

    # Compute macroscopic variables
    rho = np.sum(f, axis=2)
    u = np.sum(f * velocities[:, 0], axis=2) / rho
    v = np.sum(f * velocities[:, 1], axis=2) / rho

    # Plot the velocity magnitude every 500 iterations
    if it % 5 == 0:
        plt.clf()  # Clear the current figure to avoid overlapping
        plt.imshow(np.sqrt(u**2 + v**2).T, cmap='gnuplot')
        plt.colorbar()
        plt.title(f'Iteration {it}')
        plt.pause(0.005)

plt.show()
