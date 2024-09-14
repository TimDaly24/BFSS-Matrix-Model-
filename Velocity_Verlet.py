import numpy as np
import matplotlib.pyplot as plt

# Velocity Verlet Method
def verlet(x0, v0, step, L, k, M):
    x_values = [x0]
    v_values = [v0]

    for i in range(L):
        
        v_half = v0 + 1/2 * step * F(x0, k)
        x0 = x0 + step * v_half
        v0 = v_half + 1/2 * step *  F(x0, k)

    
        
        x_values.append(x0)
        v_values.append(v0)
        
    return np.array(x_values),np.array(v_values)

def F(x, k):
    return k * (-x)

def Hamiltonian(x, v, m, k):
    
    return 1/2 * ((v * m)**2) + 1/2 * k * (x**2)

# Metopolis-Hastings algorithim 
def Metropolis(x0, v0, xn, vn, m, k):
    alpha = min(1, (np.exp(-Hamiltonian(xn, vn, m, k)))/(np.exp(-Hamiltonian(x0, v0, m, k))))
    r = np.random.random()

    
    if r <= alpha:
        return xn, vn,alpha
    else:
        return x0, v0,alpha

# Initial conditions
x0 = 1.0
v0 = np.random.normal(0, np.sqrt(1.0))
k = 1.0
m = 1.0
L = 5000  # Number of steps
step = 0.02 * 2 * np.pi

# This stores the accepted positions and momenta
positions = [x0]
momenta = [m * v0]

# Then we run the Velocity Verlet method employing Metropolis-Hastings
for i in range(L):
    xn, vn = verlet(x0, v0, step, 1, k, m)  # We iterate through verlet only once then employ Metropolis
    x0, v0,alpha = Metropolis(x0, v0, xn[-1], vn[-1], m, k)
    print(alpha)
    positions.append(x0)
    momenta.append(m * v0)

# Plotting the phase space plot
plt.plot(positions, momenta, color='orange')
plt.xlabel('x')
plt.ylabel('p')
plt.axhline(y=0, color="black", linestyle="solid")
plt.axvline(x=0, color="black", linestyle="solid")
plt.grid(True)
plt.show()