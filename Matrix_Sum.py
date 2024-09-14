import numpy as np 
import numpy as np
import matplotlib.pyplot as plt
import time 

start_time = time.time()

# Parameters
beta = 0.5
m = 1.0   # mass
k = 1.0   # spring constant
Lambda = 5  # path length
a = beta / Lambda # Lattice Spacing
N =  5 # matrix size
iterations = 10000  # number of HMC iterations to run
burn_in = int(iterations / 10)
leap_step = 0.1



def F(X):
    force = np.zeros_like(X)  # Initialize the force array with the same shape as X
    for t in range(Lambda):
        if t == 0:
            force[0] = a *((2 / a**2 + m**2 ) * X[t] - (X[(t + 1)] + X[-1]) / (a**2))
        else:
            force[t] = a *((2 / a**2 + m**2 ) * X[t] - (X[(t + 1) % Lambda] + X[t - 1]) / (a**2))
    return force


def Leap(X, P, step):  # Leapfrog integration
    for i in range(10):
        phalf = P - step / 2 * F(X)
        X = X + step * phalf 
        P = phalf - step / 2 * F(X)
        '''for j in range(Lambda):
            x[:, j*n:(j+1)*n] = (x[:, j*n:(j+1)*n] + x[:, j*n:(j+1)*n].T) / 2'''
    return X, P

def hermite(N,Lambda):  # matrix generator
    path = []
    for _ in range(Lambda):
        A0 = np.random.normal(loc=0.0, scale=1, size=(N, N))  # random matrix
        Au = np.triu(A0)  # takes upper triangle 
        Al = np.tril(Au.T, k=-1)  # creates symmetric lower triangle
        A = Au + Al  # sums together
        path.append(A)
    return np.array(path)





def Action(X):
    part = 0
    for t in range(1,Lambda):
        if t == 0:
            part += a / 2 * np.trace((2 / (a**2) + m**2) * X[t] @ X[t] - 2 * X[t] @ (X[(t + 1) % Lambda] + X[-1]) / (a**2))
        else:
            part += a / 2 * np.trace((2 / (a**2) + m**2) * X[t] @ X[t] - 2 * X[t] @ (X[(t + 1) % Lambda] + X[t - 1]) / (a**2))
    return part

def Ham(X, P):
    ting = 0 
    for t in range(Lambda):
        ting += 1/2 * np.trace(P[t] @ P[t]) 
    return ting + Action(X)

def Metropolis(X):
    acceptance = 0
    rand_p = hermite(N, Lambda)
    x_prop, p_prop = Leap(X, rand_p, leap_step)
    Ham_current = Ham(X, rand_p)
    Ham_prop = Ham(x_prop, p_prop)
    alpha = np.exp(Ham_current - Ham_prop)
    if np.random.uniform(0, 1) <= alpha:
        acceptance = 1
        return x_prop, acceptance	
    return X, acceptance


# Initialize configuration
X = hermite(N,Lambda)



hamiltons = 0
ham_matrix = []
no_accepted = 0
S = [Action(X)]
eigen = [np.linalg.eigvals(X)]
ham_actual = m * (1/2 + (1)/(np.exp(m * beta) -1))

for i in range(1, iterations):
    X, acceptance = Metropolis(X)
    no_accepted += acceptance
    S.append(Action(X))
    eigen.append(np.linalg.eigvals(X))
    hamiltons = 0
    for j in range(Lambda):
        hamiltons += np.trace(X[j] @ X[j])
    ham_matrix.append(1/(N**2 * Lambda) * hamiltons )



constant_value = (N * N * Lambda) / 2
acc_rate = no_accepted / iterations * 100
S, iterations = S[burn_in:-1], iterations-burn_in

# Plotting the action vs. Monte Carlo iterations
plt.figure(figsize=(12, 6))
plt.plot(S, label='HMC')

plt.xlabel('Monte Carlo iterations')
plt.ylabel('<S>')
plt.axhline(y = constant_value, color= 'orange')
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
plt.xlabel('Eigenvalues')
plt.ylabel('Density')
plt.title('Histogram of Eigenvalues')
plt.show()

print('Acceptance rate: ', acc_rate, '%')
print('Expectation value of <H>: ', np.sum(ham_matrix) / iterations)
print('Expectation value should be: ', ham_actual)

print('Expectation value of <S>: ', np.sum(S) / iterations)
print('Expectation value should be: ', constant_value)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

