import numpy as np

def generate_synthetic_data(kappa=1.0, N=100, noise_std=0.0):
    Pi = np.pi
    x = np.linspace(0, 1, N)
    t = np.linspace(0, 1, N)
    X, T = np.meshgrid(x, t)

    U = np.sin(Pi * X) * np.exp(- (Pi**2) * kappa * T)

    x_flat = X.flatten().reshape(-1, 1)
    t_flat = T.flatten().reshape(-1, 1)
    u_flat = U.flatten().reshape(-1, 1)

    if noise_std > 0:
        u_flat += np.random.normal(0, noise_std, u_flat.shape)

    return x_flat, t_flat, u_flat

if __name__ == "__main__":
    x, t, u = generate_synthetic_data()
    print(x.shape, t.shape, u.shape)
