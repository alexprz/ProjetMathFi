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
    '''
        Etant donné la moyenne (previous_mean) d'un ensemble de n-1 éléments,
        renvoie la moyenne de l'ensemble constitué des n-1 éléments auxquel a été ajouté new_value.
        new_value est donc la n-ième valeur de l'ensemble.
        Utile pour calculer la moyenne au fil des itérations dans la méthode de Monte Carlo.
    '''
    return 1./n*((n-1)*previous_mean + new_value)

def compute_var(new_value, n, previous_var, previous_mean):
    '''
        Etant donné la variance (previous_var) et la moyenne (previous_mean) d'un ensemble de n-1 éléments,
        renvoie la variance de l'ensemble constitué des n-1 éléments auxquel a été ajouté new_value.
        new_value est donc la n-ième valeur de l'ensemble.
        Utile pour calculer la variance au fil des itérations dans la méthode de Monte Carlo.
    '''
    new_mean = compute_mean(new_value, n, previous_mean)
    if n == 1:
        return 0#new_value
    return (n-2)/(n-1)*previous_var+(previous_mean-new_mean)**2+1./(n-1)*(new_value-new_mean)**2

def simulate_St(dt, S0=S0, antithetic=False):
    '''
        Fonction principale sur laquelle se base toute l'étude.
        Permet de simuler une trajectoire de l'actif risqué étant donné un pas de discrétisation dt
        et un jeu de paramètres.
        Via l'attribut antithetic, elle permet de renvoyer une deuxième trajectoire : la trajectoire antithétique.
    '''

    n = int(T/dt)+1 # Nombre de tirages

    # Tirage de n valeurs selon la loi N(0, 1)
    eps = np.random.normal(size=n)

    # Calculs des accroissements multiplicateurs
    # Schéma 1
    increases = np.exp((r-(sigma**2)/2)*dt + sigma*np.sqrt(dt)*eps)
    # Schéma 2
    # increases = 1+r*dt + sigma*np.sqrt(dt)*eps
    
    increases[0] = S0

    # 1 seule trajectoire
    if not antithetic:
        return np.cumprod(increases)

    # Pour obtenir la trajectoire antithétique, il suffit de prendre l'opposé de eps
    # Calculs des accroissements multiplicateurs
    # Schéma 1
    increases_antithetic = np.exp((r-(sigma**2)/2)*dt + sigma*np.sqrt(dt)*(-eps))
    # Schéma 2
    # increases_antithetic = 1+r*dt + sigma*np.sqrt(dt)*(-eps)

    increases_antithetic[0] = S0

    # 1 trajectoire et sa trajectoire antithétique
    return np.cumprod(increases), np.cumprod(increases_antithetic)

def simulate_St_list(dt, N, S0=S0):
    '''
        Permet de simuler N strajectoires et les renvoie dans un tableau numpy
    '''
    n = int(T/dt)+1
    St_list = np.zeros((N, n))

    time0 = time()
    for i in range(N):
        St_list[i, :] = simulate_St(dt, S0=S0)

    print('Simulation time : {}'.format(time()-time0))
    return St_list

def plot_St(St, dt, H=H, K=K):
    '''
        Affiche une trajectoire donnée St
    '''
    plot_St_list([St], dt, H=H, K=K)

def plot_St_list(St_list, dt, H=H, K=K):
    '''
        Affiche len(St_list) trajectoires données dans St_list
    '''

    n = int(T/dt)+1
    X = [k*dt for k in range(n)]
    for St in St_list:
        if is_activated(St, H=H):
            plt.plot(X, St, c='green', linewidth=0.5)
        else:
            plt.plot(X, St, c='red', linewidth=0.5)


    # Barrière et Strike
    plt.plot(X, H*np.ones(n), c='red', linestyle='dashed', linewidth=2)
    plt.plot(X, K*np.ones(n), c='yellow', linestyle='dashed', linewidth=2)


    plt.xlabel('t')
    plt.ylabel('Asset price')
    plt.show()

def stats_St_list(St_list, dt, H=H, show=False):
    '''
        Étant donné un ensemble de trajectoires St_list, renvoie un dictionnaire stats
        contenant des informations comme : le prix moyen, l'écart type, le nombre d'options activées...
        Si show==True affiche les trajectoires
    '''
    stats = dict()
    stats['N'] = len(St_list)
    stats['nb_activated'] = 0
    stats['dt'] = dt
    payoffs = []

    time0 = time()
    for St in St_list:
        if is_activated(St, H=H):
            stats['nb_activated'] += 1
        payoffs.append(payoff_down_and_out(St, H=H))

    stats['computation_time'] = time()-time0
    stats['mean_payoff'] = np.mean(payoffs)
    stats['std_payoff'] = np.std(payoffs)
    prices = np.array(payoffs)*np.exp(-r*T)
    stats['mean_price'] = np.mean(prices)
    stats['std_price'] = np.std(prices)
    analytic_price = analytic_price_down_and_out4()
    stats['abs_error'] = abs(analytic_price - stats['mean_price'])
    stats['percent_activated'] = 100*stats['nb_activated']/stats['N']

    if show:
        plot_St_list(St_list, dt)

    return stats

def payoff_down_and_out(St, H=H):
    ''' 
        Calcule le payoff d'un call down and out suivant la trajectoire St
    '''
    if (St<H).any() == True:
        return 0
    return max(St[-1]-K, 0)

def is_activated(St, H=H):
    '''
        Détermine si l'option down and out suivant la trajectoire St est activée ou non
    '''
    return not (St<H).any()

def monte_carlo(dt, N_max=100000, eps=None, analytic_criterion=False, show=False, antithetic=False):
    '''
        Estime par la méthode de Monte Carlo le prix du call down and out (entre autre).

        N_max : Nombre de simulations maximal
        eps :   si eps==None, la méthode de MC s'arrête à N_max
                sinon, sert pour le critère de convergence
        analytic_criterion :
            si True, la méthode de MC s'arrête lorsque l'erreur absolue avec la valeur exacte est inférieure à eps
            si False, le critère utilisé est celui de l'intervalle asymptotique à 95% : 1.96sigma/sqrt(n) < eps
        show : permet d'afficher l'évolution de la moyenne et l'écart type en fonction du nombre d'itérations
        antithetic : si True utilise la méthode de réduction de variance par variable antithétique
    '''
    # payoffs = []
    # St_list = []
    means = []
    stds = []

    # Sert pour stocker la moyenne, l'écart type et le nombre d'itérations.
    mean_n = 0
    var_n = 0
    count = 0

    analytic_price = analytic_price_down_and_out4()

    while count < N_max:
        # Calcul du prix d'une nouvelle simulation, selon si on utilise la réduction de variance ou non
        if antithetic:
            St, St_antithetic = simulate_St(dt, antithetic=True)
            price = payoff_down_and_out(St)*np.exp(-r*T)
            price_antithetic = payoff_down_and_out(St_antithetic)*np.exp(-r*T)

            price = (price+price_antithetic)/2.
        else:
            St = simulate_St(dt, antithetic=False)
            price = payoff_down_and_out(St)*np.exp(-r*T)

        # Calcul récurssif de la moyenne et la variance
        mean_p = mean_n
        var_p = var_n
        mean_n = compute_mean(price, count+1, mean_n)
        var_n = compute_var(price, count+1, var_p, mean_p)

        # Stocké uniquement pour l'affichage
        if show:
            means.append(mean_n)
            stds.append(np.sqrt(var_n))

        count += 1

        # Calcul des différentes bornes des critères d'arrêts
        relative_error = 100*abs(mean_n-analytic_price)/analytic_price
        absolute_error = abs(mean_n-analytic_price)
        interval_bound = 1.96*np.sqrt(var_n)/np.sqrt(count)
        print('{0:} {1:} {2:.2f} {3:.2f} {4:.4f} {5:.6f}'.format(dt, count, mean_n, var_n, interval_bound, absolute_error))
        

        # Critères d'arrêt
        if analytic_criterion and eps != None and absolute_error < eps:
            break
        if not analytic_criterion and eps != None and interval_bound < eps:
            break

    # Tracé de la moyenne et de l'écart type en fonction du nombre d'itérations
    if show:
        X = range(len(means))
        plt.plot(X, means, label='Monte Carlo')
        plt.plot(X, analytic_price*np.ones(len(means)), label='Analytical')
        plt.xlabel('Nombre d\'itérations')
        plt.ylabel('Prix')
        plt.legend()
        plt.show()

        X = range(len(stds))
        plt.plot(X, stds)
        plt.xlabel('Nombre d\'itérations')
        plt.ylabel('Écart type')
        plt.show()

    # Stockage des informations dans stats
    stats = dict()
    stats['mean_payoff'] = mean_n*np.exp(r*T)
    stats['mean_price'] = mean_n
    stats['std_price'] = np.sqrt(var_n)
    analytic_price = analytic_price_down_and_out4()
    stats['abs_error'] = abs(analytic_price - stats['mean_price'])

    relative_error = 100*abs(mean_n-analytic_price)/analytic_price
    stats['relative_error'] = relative_error
    stats['nb_iterations'] = count

    return stats

def convergence_speed_function_of_dt(dt_min, dt_max, N, eps=0.01, analytic_criterion=False):
    X = np.geomspace(dt_min, dt_max, N)
    print(X)
    Y = []
    for dt in X:
        stats = monte_carlo(dt, N_max=10000000, eps=eps, analytic_criterion=analytic_criterion)
        Y.append(stats['nb_iterations'])

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


def price_function_of_dt(dt_min, dt_max, N, N_simulation=1000):
    X = np.geomspace(dt_min, dt_max, N)
    print(X)
    Y = []

    for dt in X:
        St_list = simulate_St_list(dt, N_simulation)
        stats = stats_St_list(St_list, dt)
        print(stats['mean_price'])
        Y.append(stats['mean_price'])
        print(dt)

    plt.plot(X, Y)
    plt.xlabel('dt')
    plt.ylabel('Price')
    # plt.title('Influence of dt over activation %')
    plt.show()


def error_function_of_dt(dt_min, dt_max, N, N_simulation=1000):
    X = np.geomspace(dt_min, dt_max, N)
    # X = np.linspace(dt_min, dt_max, N)
    print(X)
    Y = []

    for dt in X:
        # St_list = simulate_St_list(dt, N_simulation)
        # stats = stats_St_list(St_list, dt)
        stats = monte_carlo(dt, N_max=N_simulation)
        print(stats['abs_error'])
        Y.append(stats['abs_error'])
        print(dt)

    plt.plot(X, Y)
    plt.xlabel('dt')
    plt.ylabel('Absolute error')
    plt.xscale('log')
    # plt.title('Influence of dt over activation %')
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

# def convergence_function_of_dt(dt_min, dt_max, N, eps=0.01, analytic_criterion=False):
#     X = np.linspace(dt_min, dt_max, N)
#     Y = []

#     for dt in X:
#         mean, std, nb_iterations, _ = monte_carlo(dt, N_max=10000000, eps=eps, analytic_criterion=analytic_criterion)
#         Y.append(nb_iterations)

#     plt.plot(X, Y)
#     plt.xlabel('dt')
#     plt.ylabel('Nb itérations')
#     plt.show()

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

def analytic_price_down_and_out2(S0=S0, H=H):
    alpha_p = r + (sigma**2)/2
    alpha_m = r - (sigma**2)/2

    d1 = (np.log(S0/K) + alpha_p*T)/(sigma*np.sqrt(T))
    d2 = (np.log(S0/K) + alpha_m*T)/(sigma*np.sqrt(T))
    d7 = (np.log(H**2/(S0*K)) + alpha_p*T)/(sigma*np.sqrt(T))
    d8 = (np.log(H**2/(S0*K)) + alpha_m*T)/(sigma*np.sqrt(T))

    n1 = norm.cdf(d1)
    n2 = norm.cdf(d2)
    n7 = norm.cdf(d7)
    n8 = norm.cdf(d8)
    
    C=S0*n1-K*np.exp(-r*T)*n2-H*(S0/H)**(-2*r/sigma**2)*n7-(S0/H)**(1-2*r/sigma**2)*K*np.exp(-r*T)*n8

    return C

def analytic_price_down_and_out3(S0=S0, H=H):
    nu=np.sqrt(sigma/T)
    alpha_p = r + (nu**2)/2
    alpha_m = r - (nu**2)/2
    
    

    d1 = (np.log(S0/K) + alpha_p*T)/(nu*np.sqrt(T))
    d2 = (np.log(S0/K) + alpha_m*T)/(nu*np.sqrt(T))
    d7 = (np.log(H**2/(S0*K)) + alpha_p*T)/(nu*np.sqrt(T))
    d8 = (np.log(H**2/(S0*K)) + alpha_m*T)/(nu*np.sqrt(T))

    n1 = norm.cdf(d1)
    n2 = norm.cdf(d2)
    n7 = norm.cdf(d7)
    n8 = norm.cdf(d8)
    
    C=S0*n1-K*np.exp(-r*T)*n2-H*(S0/H)**(-2*r/sigma**2)*n7-(S0/H)**(1-2*r/sigma**2)*K*np.exp(-r*T)*n8

    return C

def analytic_price_down_and_out4(S0=S0, H=H, K=K):
    lbd = (r+(sigma**2)/2)/(sigma**2)
    x = (np.log(H**2/(S0*K))+(r-(sigma**2)/2)*T)/(sigma*np.sqrt(T))

    Cin =  S0*(H/S0)**(2*lbd)*norm.cdf(x) - K*np.exp(-r*T)*(H/S0)**(2*lbd-2)*norm.cdf(x-sigma*np.sqrt(T))
    C = analytic_call_vanilla(S0=S0, K=K)
    Cout = C - Cin

    return Cout

def analytic_call_vanilla(S0=S0, K=K):
    d1 = (np.log(S0/K) + (r + (sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    n1 = norm.cdf(d1)
    n2 = norm.cdf(d2)

    return S0*n1 - K*np.exp(-r*T)*n2

def analytic_price_function_of_H(r_min, r_max, N, dt, N_simulation=1000):
    X = np.linspace(r_min, r_max, N)
    Y1 = []
    Y2 = []
    Y3 = []

    for r in X:
        H = S0*r
        price = analytic_price_down_and_out4(S0=S0, H=H)
        print(price)
        Y1.append(price)
        Y2.append(analytic_call_vanilla(S0=S0, K=K))
        St_list = simulate_St_list(dt, N_simulation, S0=S0)
        stats = stats_St_list(St_list, dt, H=H)
        print(stats['mean_payoff'])
        Y3.append(stats['mean_payoff'])

    # plt.plot(X, Y1, label='Analytical down and out')
    plt.plot(X, Y3, label='Monte Carlo down and out', color='orange')
    plt.plot(X, Y2, label='Analytical vanilla', linestyle='--', color='green')
    plt.xlabel('H/S0')
    plt.ylabel('Price')
    # plt.title('Influence of H/S0 over mean payoff')
    plt.legend()
    plt.show()

def activation_function_of_H(r_min, r_max, N, dt, N_simulation=1000):
    X = np.linspace(r_min, r_max, N)
    Y = []

    for r in X:
        H = S0*r
        St_list = simulate_St_list(dt, N_simulation, S0=S0)
        stats = stats_St_list(St_list, dt, H=H)
        print(stats['percent_activated'])
        Y.append(stats['percent_activated'])

    plt.plot(X, Y)
    plt.xlabel('H/S0')
    plt.ylabel('Activation %')
    # plt.title('Influence of H/S0 over mean payoff')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dt = 0.001
    # St = simulate_St(dt)
    # St, St_antithetic = simulate_St(dt, antithetic=True)
    # plot_St_list([St, St_antithetic], dt)
    # print(payoff_down_and_out(St))
    # St_list = simulate_St_list(dt, 5)
    # print(stats_St_list(St_list, dt))
    # payoff_function_of_H(0, 1, 100, dt, N_simulation=100000)
    # stats_function_of_dt(1, 0.0001, 200, N_simulation=10000)
    # plot_St_list(St_list, dt)
    # print(monte_carlo(dt=0.0001, N_max=250000, show=True))
    # print(monte_carlo(dt=0.0001, N_max=1000, show=False))
    # print(monte_carlo(dt=0.1, N_max=20, show=True, eps=0.5))
    # convergence_speed_function_of_dt(0.01, 0.1, 10, eps=0.0001)
    # print(monte_carlo(dt=0.001, N_max=250000, eps=0.01, analytic_criterion=False))
    # print(monte_carlo(dt=0.01, N_max=1000000, eps=0.005))
    # convergence_function_of_dt(0.001, 0.1, 100, eps=0.01, analytic_criterion=False)
    # convergence_speed_function_of_dt(0.001, 0.1, 100, eps=0.01, analytic_criterion=False)
    # print(analytic_price_down_and_out())
    # analytic_price_function_of_H(0, 1, 100, dt, N_simulation=100000)
    # activation_function_of_H(0, 1, 100, dt, N_simulation=100000)
    # price_function_of_dt(0.00001, 0.1, 10, N_simulation=100000)
    error_function_of_dt(0.00001, 1, 10, N_simulation=10000)