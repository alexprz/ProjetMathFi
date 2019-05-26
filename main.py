import numpy as np

# Parameters
T = 1.
r = 0.03
mu = -0.05
sigma = 0.05

def simulate_St(S0, dt):
    n = int(T/dt)+1
    St = np.zeros(n)
    St[0] = S0

    for k in range(1, n):
        eps = np.random.normal()
        St[k] = St[k-1]*(1+mu*dt + np.sqrt(dt)*eps)

    return St

if __name__ == '__main__':
    print(simulate_St(10, 0.1))