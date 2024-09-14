import numpy as np
import matplotlib.pyplot as plt

# Leapfrog integration method
def leap(x0, v0, step, k, m):
    for i in range(10):
        v_half = v0 + (step / 2) * F(x0, k)
        x0 = x0 + step * v_half
        v0 = v_half + (step / 2) * F(x0, k)
    return x0, v0


# Force function for harmonic oscillator
def F(x, k):
    return -k * x

# Hamiltonian function
def Hamiltonian(x, p, m, k):
    kinetic = np.sum(p**2) / (2 * m)
    potential = np.sum(x**2)/2
    return kinetic + potential

# Metropolis-Hastings Algorithm 
def Metropolis(x0, v0, xn, vn, m, k):
    current_H = Hamiltonian(x0, v0, m, k)
    proposed_H = Hamiltonian(xn, vn, m, k)
    
    alpha = min(np.exp(current_H - proposed_H), 1)

    if np.random.uniform(0, 1) < alpha:
        return xn, vn, proposed_H,1
    
    else:
        return x0, v0, current_H,0

# Parameters
beta =45
m = 1.0     
k = 1.0     
Lambda = 100
a = beta/Lambda
leap_step = 0.7

# Initial random path
X = np.random.normal(loc=0, scale=2, size=Lambda)


steps = 3000

# Arrays to store Hamiltonians and paths
hamiltonians = []
hamiltonians.append(X)
actions = []
accept = 0
action_actual = (Lambda)/2

for i  in range(steps):
    # Propose new momentum from normal distribution
    kick = np.random.normal(loc=0, scale=1, size=Lambda)
    x_new, v_new = leap(X, kick, leap_step, k, m)
    X, V, H,acceptance = Metropolis(X, kick, x_new, v_new, m, k)
    accept += acceptance
    actions.append(Hamiltonian(X, np.zeros_like(X), m, k))
    hamiltonians.append(X)
mat = []
mat = np.concatenate(hamiltonians)
plt.plot(actions)
plt.xlabel('Monte Carlo iterations')
plt.ylabel('Action <S>')
plt.axhline(y=action_actual,color ='orange')
plt.grid(True)
plt.show()



print('Acceptance rate: ',accept/steps*100,'%')
print('Expectation value of <S>: ',np.sum(actions)/steps)
print('Expectation value of <S> should be: ', action_actual)









