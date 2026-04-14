import numpy as np
import matplotlib.pyplot as plt

x = np.array([30,35,45,50,60,70,80,90,100,120])
y = np.array([8.5,10.2,13.0,14.5,18.2,20.1,23.5,26.2,30.5,35.0])

x_mean = np.mean(x)
y_mean = np.mean(y)

x_c = x - x_mean
y_c = y - y_mean

print("Media de x:", x_mean)
print("Media de y:", y_mean)

def SSE(beta):
    return np.sum((y_c - beta * x_c)**2)

betas = np.arange(0, 1.001, 0.001)
errors = []

for b in betas:
    errors.append(SSE(b))

errors = np.array(errors)

beta_opt = betas[np.argmin(errors)]
error_min = np.min(errors)

print("\nBeta óptima:", round(beta_opt, 4))
print("Error mínimo:", round(error_min, 4))

beta_analitico = np.sum(x_c * y_c) / np.sum(x_c**2)

print("Beta analítica:", round(beta_analitico, 4))

plt.figure()

plt.scatter(x_c, y_c)

for beta in [0.1, 0.3, 0.5, 0.7, 0.9]:
    y_pred = beta * x_c
    plt.plot(x_c, y_pred, label=f'β={beta}')

plt.title("Comparativa de Pendientes")
plt.xlabel("x centrado")
plt.ylabel("y centrado")
plt.legend()
plt.grid()

plt.show()

plt.figure()

plt.plot(betas, errors)
plt.axvline(beta_opt, linestyle='--', label=f'Óptimo = {beta_opt:.3f}')

plt.title("Paisaje del Error (SSE)")
plt.xlabel("β")
plt.ylabel("SSE")
plt.legend()
plt.grid()

plt.show()

print("\nINTERPRETACIÓN:")
print(f"La pendiente óptima β1 ≈ {beta_opt:.3f} indica cuánto aumenta la renta (en miles)")
print("por cada metro cuadrado adicional.")
print("La gráfica del SSE muestra que la función es convexa y tiene un único mínimo.")x