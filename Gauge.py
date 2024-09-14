import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Parameters
beta = 0.17
m = 1.0  # mass
Lambda = 5  # path length
a = beta / Lambda  # Lattice Spacing
N = 7 # matrix size
iterations = 10000  # number of HMC iterations to run
burn_in = int(iterations / 10)
leap_step = 0.01
g = np.identity(N)
g_inv=inv(g)




def F(X):
    const = 1 + (a**2*m**2)/(2)
    initial = 1/a * (X[1] + inv(g) @ X[-1] @ g) -2/a *const* X[0]
    final = 1/a * (X[Lambda -2] + g @ X[0] @ inv(g)) -2/a *const* X[-1]
    
    F = [initial]
    for n in range(1,Lambda -1):
        F.append(1/a * (X[n-1] + X[n+1]) - 2/a * const * X[n])
    F.append(final)
    return np.array(F)

def Leap(X, P, step):  # Leapfrog integration
    for i in range(5):
        phalf = P + step / 2 * F(X)
        X = X + step * phalf
        P = phalf + step / 2 * F(X)
    for j in range(Lambda):
        X[j] = (X[j] + X[j].T) / 2
    return X, P


def hermite(N, Lambda,scal):
    path = []
    for _ in range(Lambda):
        A0 = np.random.normal(loc=0.0, scale=scal, size=(N, N))  # random matrix
        Au = np.triu(A0)  # takes upper triangle 
        Al = np.tril(Au.T, k=-1)  # creates symmetric lower triangle
        A = Au + Al  # sums together
        path.append(A)
    return np.array(path)


def Action(X):
    part1 = np.zeros_like(X[0])
    part2 = 0
    for n in range(Lambda - 1):
        part1 += (X[n] @ X[n+1])
        part2 += np.trace((1 + (a**2*beta**2*m**2)/(2))* (X[n] @ X[n]))
    part2 += np.trace((1 + (a**2*beta**2*m**2)/(2))* (X[Lambda-1] @ X[Lambda-1]))
    action = 1/a * (-np.trace(part1 + (X[Lambda-1] @ g @ X[0] @ inv(g))) + part2)
    return action

def action(x, g=g, g_inv=g_inv):
    Lambda = len(x)
    A1 = 0
    A2 = 0             # will use these to break up the calculation
    c = (a*a*m*m)/2 # defining this constant to be used later
    
    
    for i in range(Lambda-1):
        A1 += x[i]@x[i+1]
    for i in range(Lambda):    
        A2 += np.trace((1 + c)*x[i]@x[i])
    A1 += x[-1] @ g @ x[0] @ g_inv
    
    return (-np.trace(A1) + A2)/a


def Ham(X, P):
    kinetic = 0
    for i in range(Lambda):
        kinetic += 0.5 * np.trace(P[i] @ P[i])
    return kinetic + Action(X)


def Metropolis(X):
    acceptance = 0
    kick = hermite(N, Lambda,1)
    X_prop, P_prop = Leap(X, kick, leap_step)
    Ham_current = Ham(X, kick)
    Ham_prop = Ham(X_prop, P_prop)
    alpha = np.exp(Ham_current - Ham_prop)
    if np.random.uniform(0, 1) <= alpha:
        acceptance = 1
        return X_prop,P_prop, acceptance
    return X,kick, acceptance


# Initialize configuration
X = hermite(N, Lambda,1)
for n in range(Lambda):
    X[n] = 1/5*X[n]

hamiltons = 0
ham_matrix = []
no_accepted = 0
S = [Action(X)]
eigen=[]
ham_actual = m * (1 / 2 + (1) / (np.exp(m * beta) - 1))

for i in range(1, iterations):
    X,P, acceptance = Metropolis(X)
    no_accepted += acceptance
    S.append(Action(X))
    hamiltons = 0
    eigen.append(np.linalg.eigvals(X))
    for j in range(Lambda):
        hamiltons += np.trace(X[j] @ X[j])
    ham_matrix.append(1 / (N ** 2 * Lambda) * hamiltons)

expect_S = (N * N * Lambda) / 2

acc_rate = no_accepted / iterations * 100
#S, iterations = S[burn_in:], iterations - burn_in

# Plotting the action vs. Monte Carlo iterations
plt.figure(figsize=(12, 6))
plt.plot(S, label='HMC')
plt.xlabel('Monte Carlo iterations')
plt.ylabel('<S>')
plt.axhline(y=expect_S, color='orange')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(ham_matrix)
plt.xlabel('Monte Carlo iterations')
plt.ylabel('Hamiltonian <H>')
plt.axhline(y=ham_actual, color='orange')
plt.grid(True)
plt.show()

eigen = np.ndarray.flatten(np.array(eigen))
plt.figure()
plt.hist(eigen, 30, density=True)
plt.show()
print('Acceptance rate: ', acc_rate, '%')
print('Expectation value of <H>: ', np.sum(ham_matrix) / iterations)
print('Expectation value should be: ', ham_actual)
print('Expectation value of <S>: ', np.sum(S) / iterations)
print('Expectation value should be: ', expect_S)

    
    



    
    


