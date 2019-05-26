import numpy as np
from matplotlib import pyplot as plt

# Parameters
T = 1.
r = 0.03
mu = -0.05
sigma = 0.05
S0 = 10
H = 0.9*S0
K = 0.8*S0

def simulate_St(S0, dt):
    n = int(T/dt)+1
    St = np.zeros(n)
    St[0] = S0

    for k in range(1, n):
        eps = np.random.normal()
        St[k] = St[k-1]*(1+mu*dt + np.sqrt(dt)*eps)

    return St

def plot_St(S0, dt, N):

    X = [k*dt for k in range(int(T/dt)+1)]
    for k in range(N):
        St = simulate_St(S0, dt)
        plt.plot(X, St)


    plt.xlabel('t')
    plt.ylabel('Asset price')
    plt.show()

def payoff_down_and_out(St):
    n, = St.shape
    for k in range(n):
        if St[k] < H:
            return 0

    return max(St[-1]-K, 0)


if __name__ == '__main__':
    St = simulate_St(10, 0.1)
    print(payoff_down_and_out(St))
    # plot_St(S0, 0.1, 100)