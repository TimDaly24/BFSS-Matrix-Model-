import numpy as np
import matplotlib.pyplot as plt

def leap(x0, v0, step, k):
    for _ in range(1):
        v_half = v0 - step/2* F(x0,k)
        x0 = x0 + step * v_half #sum
        v0 = v_half - step/2 * F(x0, k)
    return x0, v0

def F(x, k):
    return k * x

def Hamiltonian(x, v, m, k):
    return 0.5 * m * v**2 + 0.5 * k * x**2

# Metropolis-Hastings Algorithm 
def Metropolis(x0, v0, xn, vn, m, k):
    current_H = Hamiltonian(x0, v0, m, k)
    proposed_H = Hamiltonian(xn, vn, m, k)
    
    alpha = min(1, np.exp(current_H - proposed_H))
    r = np.random.random()

    if r <= alpha:
        return xn,vn, proposed_H
    else:
        return x0,v0, current_H


# Initial conditions
x0 = 0.0

k = 1.0
m = 1.0
L = 1000  # Number of steps
step = 0.1  # Step size
positions = []
momenta = []
action = []

for _ in range(L):
    # Propose new momentum from normal distribution
    kick = np.random.normal(0,1)
    xn, vn = leap(x0, kick, step, k)
    x0,v0,H= Metropolis(x0, kick,xn, vn, m, k)
    action.append(Hamiltonian(x0, 0,m,k))
    positions.append(x0)
    momenta.append(v0)



# Define the circle parameters
radius = 2
center = (0, 0)

# Create an array of angles from 0 to 2*pi
theta = np.linspace(0, 2 * np.pi, 100)

# Calculate the x and y coordinates of the circle
x = center[0] + radius * np.cos(theta)
y = center[1] + radius * np.sin(theta)

# Plot the circle
plt.figure(figsize=(6, 6))
plt.plot(x, y, label=f'Circle with radius {radius}')



# Add labels and a title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Circle Plot')
plt.legend()



# Plotting the phase space plot
plt.scatter(positions, momenta)
plt.xlabel('x')
plt.ylabel('p')
plt.axhline(y=0, color="black", linestyle="solid")
plt.axvline(x=0, color="black", linestyle="solid")
plt.grid(True)
plt.show()

'''ar= []
for i in range(L):
    ar.append(i)
plt.plot(ar,action)
plt.show()'''
exps = 0
for i in action:
    exps += i

print(exps/L)





