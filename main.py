import numpy as np
from matplotlib import pyplot as plt

# Parameters
T = 1.
r = 0.1
mu = -0.05
sigma = 0.2
S0 = 50
H = 40
K = 50

def compute_mean(new_value, n, previous_mean):
    return 1./n*((n-1)*previous_mean + new_value)

def compute_std(new_value, n, previous_std, previous_mean):
    new_mean = compute_mean(new_value, n, previous_mean)
    if n == 1:
        return new_value
    return (n-2)/(n-1)*previous_std+(previous_mean-new_mean)**2+1./(n-1)*(new_value-new_mean)**2

def simulate_St(dt):
    n = int(T/dt)+1
    # St = np.zeros(n)
    # St[0] = S0

    # for k in range(1, n):
    #     eps = np.random.normal()
    #     St[k] = St[k-1]*(1+mu*dt + np.sqrt(dt)*eps)

    # return St

    eps = np.random.normal(size=n)
    increases = 1+mu*dt + np.sqrt(dt)*eps
    increases[0] = S0

    return np.cumprod(increases)

def plot_St(dt, N):

    X = [k*dt for k in range(int(T/dt)+1)]
    for k in range(N):
        St = simulate_St(dt)
        plt.plot(X, St)


    plt.xlabel('t')
    plt.ylabel('Asset price')
    plt.show()

def plot_St_list(St_list, dt):
    n = int(T/dt)+1
    X = [k*dt for k in range(n)]
    for St in St_list:
        if is_activated(St):
            plt.plot(X, St, c='green', linewidth=0.5)
        else:
            plt.plot(X, St, c='red', linewidth=0.5)


    plt.plot(X, H*np.ones(n), c='red', linestyle='dashed', linewidth=2)
    plt.plot(X, K*np.ones(n), c='yellow', linestyle='dashed', linewidth=2)


    plt.xlabel('t')
    plt.ylabel('Asset price')
    plt.show()

# def payoff_down_and_out(St):
#     n, = St.shape
#     for k in range(n):
#         if St[k] < H:
#             return 0

#     return max(St[-1]-K, 0)

def payoff_down_and_out(St):
    return int(not (St<H).any() == True)*max(St[-1]-K, 0)
    for k in range(n):
        if St[k] < H:
            return 0

    return max(St[-1]-K, 0)

def is_activated(St):
    n, = St.shape
    for k in range(n):
        if St[k] < H:
            return False
    return True

def monte_carlo(dt, N_max=100000, N_min=200, eps=None, show=False):
    payoffs = []
    St_list = []
    mean_n = -1
    std_n = 0
    count = 0

    while count < N_max:
        St = simulate_St(dt)
        St_list.append(St)
        payoff = payoff_down_and_out(St)
        payoffs.append(payoff)
        mean_p = mean_n
        std_p = std_n
        # mean_n = 1/(count+1)*(count*mean_n + payoff)
        mean_n = compute_mean(payoff, count+1, mean_n)
        std_n = compute_std(payoff, count+1, std_p, mean_p)
        var = np.var(payoffs, ddof=1)

        count += 1
        # if count >= N_min and abs(mean_n - mean_p) < eps:
        print('{} {} {}'.format(mean_n, std_n, 1.96*np.sqrt(std_n)/np.sqrt(count)))
        # if count >= N_min and 1.96*np.sqrt(std_n)/np.sqrt(count) < eps:
        if count >= N_min and eps != None and abs(mean_n - mean_p) < eps:

            # print('Delta mean : {}'.format(abs(mean_n - mean_p)))
            print('Error : {}'.format(1.96*np.sqrt(std_n)/np.sqrt(count)))
            break

    if show:
        plot_St_list(St_list, dt)

    return mean_n, count

def convergence_speed_function_of_dt(dt_min, dt_max, N, eps):
    X = np.linspace(dt_min, dt_max, N)
    Y = []
    for dt in X:
        Y.append(monte_carlo(dt, eps=eps)[1])

    plt.plot(X, Y)
    plt.xlabel('dt')
    plt.ylabel('Nb iterations')
    plt.show()



if __name__ == '__main__':
    dt = 0.001
    St = simulate_St(dt)
    # print(payoff_down_and_out(St))
    # plot_St_list([St], dt)
    # print(monte_carlo(dt=0.1, N_max=1000, show=False, eps=0.0001))
    print(monte_carlo(dt=0.0001, N_max=1000, show=True))
    # print(monte_carlo(dt=0.1, N_max=20, show=True, eps=0.5))
    # convergence_speed_function_of_dt(0.01, 0.1, 10, eps=0.0001)