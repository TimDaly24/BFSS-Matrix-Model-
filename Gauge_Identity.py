import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
start_time = datetime.now()


iterations = 10000
burn_in = int(iterations/10)
Lambda = 5
N = 7
beta = 0.3
a = beta/Lambda
m = 1.0
g = -np.eye(N)
g_inv = np.linalg.inv(g)
eps = 0.05





def Hermitise(C):
    tri1 = np.triu(C) - np.diag(np.diag(C))
    tri2 = np.triu(C).T.conj()
    C = tri1 + tri2
    return C

def get_Hermitian(Lambda,N):
    path = [] 
    for _ in range(Lambda):
        D = np.diag(np.random.randn(N))
        C = np.random.normal(0, 1/2, size=(N, N))
        B = np.random.normal(0, 1/2, size=(N, N))
        B = 1j*(B - np.diag(B))
        C = C - np.diag(C)
        C = B+C+D
        C_sym = Hermitise(C)
        path.append(C_sym)
    return path



def action(x):
    Lambda = len(x)
    A1 = 0
    A2 = 0             # will use these to break up the calculation
    c = 2+a*a*m*m # defining this constant to be used later
    
    for i in range(Lambda-1):
        A1 += x[i]@x[i+1]
        A2 += np.trace(c*x[i]@x[i])
    for i in range(1, Lambda):
        A1 += x[i]@x[i-1]
    A2 += np.trace(c*x[-1]@x[-1])
    A1 += x[-1] @ g @ x[0] @ g_inv + x[0] @ g_inv @ x[-1] @ g
    # derived from non-gauged version so g&g_inv may be in wrong order here, I guessed it
    
    return np.real(-np.trace(A1) + A2)/(2*a)


def F(x):
    c = (2/a + a*m*m)
    F = [1/a * (x[1] + g_inv@x[-1]@g) - c*x[0]] # first term -dS/dx_1
    
    Lambda = len(x)
    for i in range(1, Lambda-1):
        A = 1/a * (x[i-1] + x[i+1]) - c*x[i]
        F.append(A) # gathers all middle terms 
    
    F.append(1/a * (x[-2] + g@x[0]@g_inv) - c*x[-1])
    
    return np.array(F)


def Leap(X, P, space=eps): # Leapfrog integration
    x=X.copy()
    p=P.copy()
    phalf = np.zeros_like(x)
    for i in range(10):
        phalf = p + space/2*F(x)
        x = x + space*phalf
        p = phalf + space/2*F(x)

    for i in range(Lambda):
        x[i] = (x[i]+x[i].conj().T)/2
        p[i] = (p[i]+p[i].conj().T)/2
    return x, p


def Ham(x, p):
    Lambda = len(x)
    n = len(x[0])
    k = np.zeros((N,N), dtype=np.complex128)
    for i in range(Lambda):
        k += p[i]@p[i]
    kinetic = 0.5*np.trace(k) #over m ?     

    return kinetic + action(x)


def Metropolis(x):

    acceptance = 0
    rand_p = get_Hermitian(Lambda, N)
    x_prop, p_prop = Leap(x, rand_p)
    Ham_current = Ham(x, rand_p)
    Ham_prop = Ham(x_prop, p_prop)
    alpha = np.exp(Ham_current - Ham_prop)
    # alpha is probability of accepting new configuration if alpha<1
    if np.random.uniform(0,1) <= alpha:
        acceptance = 1
        return x_prop, acceptance	
    #decides whether or not to update the configuartion and tracks acceptance
    
    return x, acceptance




x = get_Hermitian(Lambda, N)
no_accepted=0
S = [action(x)]
eigen = [np.linalg.eigvals(x)]
hamiltons = 0
ham_matrix = []
ham_actual = m * (1 / 2 + (1) / (np.exp(m * beta) - 1))

def eig(x):
    return sum(np.ndarray.flatten(abs(np.linalg.eigvals(x))))

for i in range(1, iterations):
    # Propose new momenta from normal distribution
    x, acceptance = Metropolis(x)
    no_accepted += acceptance
    S.append(action(x))
    hamiltons = 0
    eigen.append(np.real(np.linalg.eigvals(x)))
    for j in range(Lambda):
        hamiltons += np.trace(x[j] @ x[j])
    ham_matrix.append(1 / (N ** 2 * Lambda) * hamiltons)
    
acc_rate = no_accepted / iterations * 100
S, iterations = S[burn_in:-1], iterations-burn_in


# Plotting the action vs. Monte Carlo iterations
plt.plot(S)
plt.xlabel('Monte Carlo iterations')
plt.ylabel('Action <S>')
plt.axhline(y = (N*N * Lambda)/2, color="red")
plt.grid(True)
plt.show()

plt.plot(ham_matrix)
plt.xlabel('Monte Carlo iterations')
plt.ylabel('Hamiltonian <H>')
plt.axhline(y=ham_actual, color='orange')
plt.grid(True)
plt.show()

print('Acceptance rate: ', acc_rate, '%')
print('Expectation value of <S>: ', np.sum(S)/(iterations))
print('Expectation value should be: ', (N*N * Lambda) / 2)
print('Expectation value of <H>: ', np.real((np.sum(ham_matrix))) / iterations)
print('Expectation value should be: ', ham_actual)



eigen = np.ndarray.flatten(np.array(eigen))
plt.figure()
plt.hist(eigen, 30, density=True)
# do your work here
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))




