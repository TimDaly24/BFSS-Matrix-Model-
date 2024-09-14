import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
start_time = datetime.now()

plt.style.use('bmh')
bleu = (52/255, 207/255, 235/255)  
greeen = (165/255, 235/255, 52/255)

iterations = 10000
burn_in = 0#int(iterations/5 )
Lambda = 5
n = 25
beta = 0.3
a = beta/Lambda
m = 1.0
g = -np.eye(n)
g_inv = np.linalg.inv(g)
eps = 0.037

def norm_sym(n, scale): # random symmetric matrix generator
    """
    Function to generate random n*n symmetric matrix with entries chosen from
    normal distribution
    """
    A0 = np.random.normal(loc=0.0, scale=scale, size=(n, n)) # random matrix
    Au = np.triu(A0) # takes upper triangle 
    Al = np.tril(Au.T, k=-1) # creates symmetric lower triangle
    A = Au+Al # sums together
    return A


def mat_array(Lambda, n, scale):   # generates arrays of matrices
    x = []
    for i in range(Lambda):
        x.append(norm_sym(n, scale))
    return x


def action(x, g=g, g_inv=g_inv):
    Lambda = len(x)
    A1 = 0
    A2 = 0             # will use these to break up the calculation
    c = 2+a*a*m*m # defining this constant to be used later
    
    
    for i in range(Lambda-1):
        A1 += x[i]@x[i+1]
    for i in range(1, Lambda):
        A1 += x[i]@x[i-1]
    for i in range(Lambda):    
        A2 += np.trace(c*x[i]@x[i])
    A1 += x[-1] @ g @ x[0] @ g_inv + x[0] @ g_inv @ x[-1] @ g
    # derived from non-gauged version so g&g_inv may be in wrong order here, I guessed it
    
    return (-np.trace(A1) + A2)/(2*a)


def Action_Matrix(x, a):
    N = len(x)   # size of square matrices comprising x
    Lambda = int(len(x[0])/n)      # number of matrices in  x array
    M = np.zeros((Lambda * N, Lambda * N))

    for i in range(Lambda):
        M[i*N:(i+1)*N, i*N:(i+1)*N] = (2 + a*a*m*m)*np.eye(N)
        if i != Lambda - 1:
            M[i*N:(i+1)*N, (i+1)*N:(i+2)*N] = -np.eye(N)
            M[(i+1)*N:(i+2)*N, i*N:(i+1)*N] = -np.eye(N)
    M[0:N, (Lambda-1)*N:Lambda*N] += -np.eye(N)
    M[(Lambda-1)*N:Lambda*N, 0:N] += -np.eye(N)
    return M


def Action(x, a):   # I will do this piecewise and then sum the pieces
	"""
    Gives the action for a given list of position matrices
    """
	V1 = Action_Matrix(x, a)
	Action = 1/(2*a)*np.trace((x @ V1 @ x.T))
	return Action


def F(x, g=g, g_inv=g_inv):
    c = (2/a + a*m*m)
    F = [1/a * (x[1] + g_inv@x[-1]@g) - c*x[0]] # first term -dS/dx_1
    
    Lambda = len(x)
    for i in range(1, Lambda-1):
        A = 1/a * (x[i-1] + x[i+1]) - c*x[i]
        F.append(A) # gathers all middle terms 
    
    F.append(1/a * (x[-2] + g@x[0]@g_inv) - c*x[-1])
    
    return F


def Leap(X, P, space=eps, N=5): # Leapfrog integration
    x=X.copy()
    p=P.copy()
    phalf = x.copy()
    for i in range(N):
        for i in range(Lambda):
            phalf[i] = p[i] + space/2*F(x)[i]
            x[i] = x[i] + space*phalf[i]
            p[i] = phalf[i] + space/2*F(x)[i]
# I have confirmed that this is equivalent to the block verion
# still seams to be inflating the eigenvalues?
            """
            phalf = p + space/2*F(x)
            x = x + space*phalf
            p = phalf + space/2*F(x)
            """
    for i in range(Lambda):
        x[i] = (x[i]+x[i].T)/2
        p[i] = (p[i]+p[i].T)/2
    return x, p


def Ham(x, p):
    Lambda = len(x)
    n = len(x[0])
    k = np.zeros((n,n))
    for i in range(Lambda):
        k += p[i]@p[i]
    kinetic = 0.5*np.trace(k) #over m ?     

    return kinetic + action(x)


def Metropolis(x):

    acceptance = 0
    rand_p = mat_array(Lambda, n, 1)
    x_prop, p_prop = Leap(x, rand_p)
    Ham_current = Ham(x, rand_p)
    Ham_prop = Ham(x_prop, p_prop)
    alpha = np.exp(Ham_current - Ham_prop)
    # alpha is probability of accepting new configuration if alpha<1
    if np.random.uniform(0,1) <= alpha:
        acceptance = 1
        return x_prop, acceptance	
    # decides whether or not to update the configuartion and tracks acceptance
    
    return x, acceptance


def check(x):
    Lambda = len(x)
    n = len(x[0])
    for k in range(Lambda): # matrix by matrix
        for i in range (n):
            for j in range(n):
                if x[k][i][j] != x[k][j][i]:
                    return 1
    return 0

x = mat_array(Lambda, n, 0.18)
no_accepted=0
S = [action(x)]
eigen = [np.linalg.eigvals(x)]
f = 0

def eig(x):
    return sum(np.ndarray.flatten(abs(np.linalg.eigvals(x))))

for i in range(1, iterations):
    # Propose new momenta from normal distribution
    x, acceptance = Metropolis(x)
    if check(x) > 0:
        print(i)
    no_accepted += acceptance
    S.append(action(x))
    #adapt_count += acceptance
    eigen.append(np.linalg.eigvals(x))
    f = i
    """
    if i % 100 == 0:
        acc_rate = adapt_count/100    
        if acc_rate < 0.7:
        	a = a/1.01
        else:
            a = a*1.01
    adapt_count = 0
    """

acc_rate = no_accepted / iterations * 100
S, iterations = S[burn_in:-1], iterations-burn_in


# Plotting the action vs. Monte Carlo iterations
plt.plot(S,label ='Action')
plt.xlabel('Monte Carlo iterations')
plt.ylabel('<S>')
plt.axhline(y = (n*n * Lambda)/2, color="orange",label='Analytic')
plt.legend()
plt.grid(True)
plt.show()



print('Acceptance rate: ', acc_rate, '%')
print('Expectation value of <S>: ', np.sum(S)/(iterations))
print('Acceptance rate should be 70%')
print('Expectation value should be: ', (n*n * Lambda) / 2)

def wigner(R, x):
    return (2/(np.pi * R*R)) * np.sqrt(R*R - x*x)


eigen = np.ndarray.flatten(np.array(eigen))
plt.figure()
plt.hist(eigen, 100, density=True)
plt.title('Normalised Histogram of Eigenvalues')
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\rho(\lambda)$')
plt.show()

print("Should be zero -->", np.sum(abs(eigen.imag)))
 
# do your work here
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
