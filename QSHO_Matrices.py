import numpy as np
import matplotlib.pyplot as plt

# Define the force as the gradient of the action
def F(X, N, a, m, Lambda):
    mid = M(N, a, m, Lambda)
    return -((a) * ( X * mid ))

# Defines the middle tridiagonal matrix
def M(N, a, m, Lambda):
    mid = tri(Lambda * N, a, m)
    mid[(Lambda * N) - 1, 0] = -1 / (a * a)
    mid[0, (Lambda * N) - 1] = -1 / (a * a)
    return np.matrix(mid)

# Leapfrog integration method in matrix formalism
def leap(x0, v0, step, m, Lambda, N):
    for _ in range(20):
        v_half = v0 + ((step / 2) * F(x0, N, step, m, Lambda))
        x0 = x0 + step * v_half
        v0 = v_half + ((step / 2) * F(x0, N, step, m, Lambda))
    return x0, v0

# Metropolis-Hastings Algorithm
def Metropolis(X, P, X_new, P_new, N, Lambda, a, m):
    current_H = Hamiltonian(X, P, N, Lambda, a, m)
    proposed_H = Hamiltonian(X_new, P_new, N, Lambda, a, m)
    alpha = min(np.exp(current_H - proposed_H), 1)
    if np.random.uniform(0, 1) < alpha:
        return X_new, P_new, proposed_H, 1
    else:
        return X, P, current_H, 0

def Hamiltonian(X, P, N, Lambda, a, m):
    kinetic = 0.5 * np.trace(P @ P.T)
    return kinetic + action(X, a, m, Lambda, N) # Calls action function 

# Defines the action
def action(X, a, m, Lambda, N):
    return (a / 2) * np.trace(X @ M(N, a, m, Lambda) @ X.T)

# Returns a tri-diagonal symmetric matrix
def tri(n, a, m):
    first = 2 / (a * a) + m * m
    b = -1 / (a * a)
    T = np.zeros((n, n))
    np.fill_diagonal(T, first)
    np.fill_diagonal(T[0:-1, 1:], b)
    np.fill_diagonal(T[1:, 0:-1], b)
    return T

# Returns a block matrix of random paths defined by Hermitian matrices 
def hermite(N, Lambda, scal):
    path = []
    for _ in range(Lambda):
        A0 = np.random.normal(loc=0.0, scale=scal, size=(N, N))  # random matrix
        Au = np.triu(A0)  # takes upper triangle
        Al = np.tril(Au.T, k=-1)  # creates symmetric lower triangle
        A = Au + Al  # sums together
        path.append(A)
    return np.block(path)

# My attempt at an adaptive step size
def adapt(step, accep, target=0.70, adapt_factor=0.001):
    if accep < target:
        step *= (1 - adapt_factor)
    else:
        step *= (1 + adapt_factor)
    return step

def extract_block(X, N, Lambda):
    blocks = []
    i = 0
    for _ in range(Lambda): 
        blocks.append(X[0:N, i:i+N])
        i += N
    return blocks 

def eigenvalues(positions):
    return np.linalg.eigvals(positions)

def wigner(R, x):
    return (2 / (np.pi * R**2)) * np.sqrt(R**2 - x**2)


# Define constants
beta = 0.1    
m = 1.0  # Mass
Lambda = 5 # Number of points
a = beta / Lambda # Initial time step
N = 6  # Size of matrices
steps = 10000  # Number of Monte Carlo steps
burn_in = 1000  # Burn-in period

expectation = (N**2 * Lambda) / 2 # Analytic expectation value


# Initial random paths
X = hermite(N, Lambda, 1)
X = X * (1/5)  # Factor here to stop it starting too high

# Arrays to store Hamiltonians and paths
hamiltonians = []                    
actions = []
eigvals = []
positions = []
summand = []
accept = 0

for i in range(steps):
    # Propose new momenta from normal distribution
    P = hermite(N, Lambda, 1)
    X_new, P_new = leap(X, P, a, m, Lambda, N)
    X, P, H, accepted = Metropolis(X, P, X_new, P_new, N, Lambda, a, m)
    actions.append(action(X, a, m, Lambda, N))
    accept += accepted
    hamiltonians.append(H)
    
    path = extract_block(X, N, Lambda)
    
    for i in range(Lambda):
        current_eigvals = np.linalg.eigvals(path[i])
        for val in current_eigvals:
            eigvals.append(val)

    # Update step size 
    if (i + 1) % 100 == 0:
        acceptance = accept / (i + 1)
        a = adapt(a, acceptance)

# Convert eigvals to a NumPy array for easier manipulation
eigvals = np.array(eigvals)
plt.hist(eigvals,density= True,bins=30)
plt.show()

plt.plot(actions,color= 'orange')
plt.axhline(y=expectation, color='blue', linestyle='-')
plt.show()

print('Acceptance rate: ', accept / steps * 100, '%')
print('Expectation value should be: ', expectation)
print('Expectation value of <S>: ', np.sum(actions) / steps)