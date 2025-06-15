import numpy as np
import matplotlib.pyplot as plt

# Parametri
kappa = 0.1
x = np.linspace(0, 1, 100)
t_vals = [0.0, 0.25, 0.5, 0.75, 1.0]

# Soluția analitică
plt.figure()
for t in t_vals:
    u = np.sin(np.pi * x) * np.exp(-np.pi**2 * kappa * t)
    plt.plot(x, u, label=f't = {t}')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('Soluția analitică pentru diferite t')
plt.legend()
plt.grid()
plt.savefig('analytic_solution_xt.png')

# Evoluția în timp în punct fix
t = np.linspace(0, 1, 100)
u_fixed = np.sin(np.pi * 0.5) * np.exp(-np.pi**2 * kappa * t)
plt.figure()
plt.plot(t, u_fixed, 'r')
plt.xlabel('t')
plt.ylabel('u(0.5,t)')
plt.title('Temperatura în x=0.5 în funcție de timp')
plt.grid()
plt.savefig('analytic_solution_t.png')
