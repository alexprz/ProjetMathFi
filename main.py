import numpy as np
from matplotlib import pyplot as plt
from time import time
from scipy.stats import norm

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

def simulate_St(dt, S0=S0, antithetic=False):
    n = int(T/dt)+1
    # St = np.zeros(n)
    # St[0] = S0

    # for k in range(1, n):
    #     eps = np.random.normal()
    #     St[k] = St[k-1]*(1+mu*dt + np.sqrt(dt)*eps)

    # return St

    eps = np.random.normal(size=n)
    # increases = 1+mu*dt + sigma*np.sqrt(dt)*eps
    increases = np.exp((r-(sigma**2)/2)*dt + sigma*np.sqrt(dt)*eps)
    increases[0] = S0

    if not antithetic:
        return np.cumprod(increases)

    # increases_antithetic = 1+mu*dt + sigma*np.sqrt(dt)*(-eps)
    increases_antithetic = np.exp((r-(sigma**2)/2)*dt + sigma*np.sqrt(dt)*(-eps))
    increases_antithetic[0] = S0

    return np.cumprod(increases), np.cumprod(increases_antithetic)

def simulate_St_list(dt, N, S0=S0):
    n = int(T/dt)+1
    St_list = np.zeros((N, n))

    time0 = time()
    for i in range(N):
        St_list[i, :] = simulate_St(dt, S0=S0)

    print('Simulation time : {}'.format(time()-time0))
    return St_list

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

def stats_St_list(St_list, dt, H=H, show=False):
    stats = dict()
    stats['N'] = len(St_list)
    stats['nb_activated'] = 0
    stats['dt'] = dt
    payoffs = []

    time0 = time()
    for St in St_list:
        if is_activated(St):
            stats['nb_activated'] += 1
        payoffs.append(payoff_down_and_out(St, H=H))

    stats['computation_time'] = time()-time0
    stats['mean_payoff'] = np.mean(payoffs)
    stats['std_payoff'] = np.std(payoffs)
    stats['percent_activated'] = 100*stats['nb_activated']/stats['N']

    if show:
        plot_St_list(St_list, dt)

    return stats

def payoff_down_and_out(St, H=H):
    if (St<H).any() == True:
        return 0
    return max(St[-1]-K, 0)*np.exp(-r*T)

def is_activated(St, H=H):
    return not (St<H).any()

def monte_carlo(dt, N_max=100000, N_min=200, eps=None):
    # payoffs = []
    # St_list = []
    mean_n = -1
    std_n = 0
    count = 0

    while count < N_max:
        St, St_antithetic = simulate_St(dt, antithetic=True)
        # St_list.append(St)
        payoff = payoff_down_and_out(St)
        payoff_antithetic = payoff_down_and_out(St_antithetic)
        # payoffs.append(payoff)
        # mean_n = 1/(count+1)*(count*mean_n + payoff)
        mean_p = mean_n
        std_p = std_n
        # mean_n = compute_mean(payoff, count+1, mean_n)
        # std_n = compute_std(payoff, count+1, std_p, mean_p)
        mean_n = compute_mean((payoff+payoff_antithetic)/2., count+1, mean_n)
        std_n = compute_std((payoff+payoff_antithetic)/2., count+1, std_p, mean_p)
        # count += 1
        # mean_p = mean_n
        # std_p = std_n
        # mean_n = compute_mean(payoff_antithetic, count+1, mean_n)
        # std_n = compute_std(payoff_antithetic, count+1, std_p, mean_p)
        # var = np.var(payoffs, ddof=1)
        # print('{} {}'.format(std_n, var))

        count += 1
        # if count >= N_min and abs(mean_n - mean_p) < eps:
        print('{0:} {1:.2f} {2:.2f} {3:.4f}'.format(count, mean_n, std_n, 1.96*np.sqrt(std_n)/np.sqrt(count)))
        if 1.96*np.sqrt(std_n)/np.sqrt(count) < eps:
            break
        # if count >= N_min and eps != None and abs(mean_n - mean_p) < eps:

        #     # print('Delta mean : {}'.format(abs(mean_n - mean_p)))
        #     print('Error : {}'.format(1.96*np.sqrt(std_n)/np.sqrt(count)))
        #     break
    # var = np.var(payoffs, ddof=1)
    # print('My var : {}'.format(std_n))
    # print('True var : {}'.format(var))
    # print('True var : {}'.format(np.std(payoffs)**2))
    # print('My mean : {}'.format(mean_n))
    # print('True mean : {}'.format(np.mean(payoffs)))
    # print(stats_St_list(St_list, dt))

    return mean_n, np.sqrt(std_n), count

def monte_carlo_fixed(dt, N_max=100000, S0=S0, H=H):
    St_list = simulate_St_list(dt, N_max, S0=S0)
    stats = stats_St_list(St_list, dt, H=H)

    return stats


def convergence_speed_function_of_dt(dt_min, dt_max, N, eps):
    X = np.linspace(dt_min, dt_max, N)
    Y = []
    for dt in X:
        Y.append(monte_carlo(dt, eps=eps)[1])

    plt.plot(X, Y)
    plt.xlabel('dt')
    plt.ylabel('Nb iterations')
    plt.show()

def stats_function_of_dt(dt_min, dt_max, N, N_simulation=1000):
    X = np.linspace(dt_min, dt_max, N)
    Y = []

    for dt in X:
        St_list = simulate_St_list(dt, N_simulation)
        stats = stats_St_list(St_list, dt)
        print(stats['percent_activated'])
        Y.append(stats['percent_activated'])

    plt.plot(X, Y)
    plt.xlabel('dt')
    plt.ylabel('Percent activated')
    plt.title('Influence of dt over activation %')
    plt.show()

def payoff_function_of_H(r_min, r_max, N, dt, N_simulation=1000):
    X = np.linspace(r_min, r_max, N)
    Y = []

    for r in X:
        H = S0*r
        St_list = simulate_St_list(dt, N_simulation, S0=S0)
        stats = stats_St_list(St_list, dt, H=H)
        print(stats['mean_payoff'])
        Y.append(stats['mean_payoff'])

    plt.plot(X, Y)
    plt.xlabel('H/S0')
    plt.ylabel('Mean payoff')
    # plt.title('Influence of H/S0 over mean payoff')
    plt.show()

def convergence_function_of_dt(dt_min, dt_max, N, eps=0.01):
    X = np.linspace(dt_min, dt_max, N)
    Y = []

    for dt in X:
        mean, std, nb_iterations = monte_carlo(dt, N_max=10000000, eps=eps)
        Y.append(nb_iterations)

    plt.plot(X, Y)
    plt.xlabel('dt')
    plt.ylabel('Nb itérations')
    plt.show()

def analytic_price_down_and_out(S0=S0, H=H):
    a = (H/S0)**(-1+2*r/(sigma**2))
    b = (H/S0)**(1+2*r/(sigma**2))

    alpha_p = r + (sigma**2)/2
    alpha_m = r - (sigma**2)/2

    d1 = (np.log(S0/K) + alpha_p*T)/(sigma*np.sqrt(T))
    d2 = (np.log(S0/K) + alpha_m*T)/(sigma*np.sqrt(T))
    d3 = (np.log(S0/H) + alpha_m*T)/(sigma*np.sqrt(T))
    d4 = (np.log(S0/H) + alpha_m*T)/(sigma*np.sqrt(T))
    d5 = (np.log(S0/H) - alpha_m*T)/(sigma*np.sqrt(T))
    d6 = (np.log(S0/H) - alpha_p*T)/(sigma*np.sqrt(T))
    d7 = (np.log(S0*K/(H**2)) - alpha_m*T)/(sigma*np.sqrt(T))
    d8 = (np.log(S0*K/(H**2)) - alpha_p*T)/(sigma*np.sqrt(T))

    n1 = norm.cdf(d1)
    n2 = norm.cdf(d2)
    n3 = norm.cdf(d3)
    n4 = norm.cdf(d4)
    n5 = norm.cdf(d5)
    n6 = norm.cdf(d6)
    n7 = norm.cdf(d7)
    n8 = norm.cdf(d8)

    P = K*np.exp(-r*T)*(n4-n2-a*(n7-n5))-S0*(n3-n1-b*(n8-n6))

    C = P + S0 - K*np.exp(-r*T)

    return C


if __name__ == '__main__':
    dt = 0.01
    # St = simulate_St(dt)
    # St, St_antithetic = simulate_St(dt, antithetic=True)
    # plot_St_list([St, St_antithetic], dt)
    # print(payoff_down_and_out(St))
    # St_list = simulate_St_list(dt, 100000)
    # print(stats_St_list(St_list, dt))
    # payoff_function_of_H(0, 1, 100, dt, N_simulation=100000)
    # stats_function_of_dt(1, 0.0001, 200, N_simulation=10000)
    # plot_St_list(St_list, dt)
    # print(monte_carlo(dt=0.0001, N_max=10, show=True, eps=0.0001))
    # print(monte_carlo(dt=0.0001, N_max=1000, show=False))
    # print(monte_carlo(dt=0.1, N_max=20, show=True, eps=0.5))
    # convergence_speed_function_of_dt(0.01, 0.1, 10, eps=0.0001)
    # print(monte_carlo(dt=0.001, N_max=250000))
    # print(monte_carlo(dt=0.01, N_max=1000000, eps=0.005))
    # convergence_function_of_dt(0.1, 0.001, 4, eps=0.01)
    print(analytic_price_down_and_out())