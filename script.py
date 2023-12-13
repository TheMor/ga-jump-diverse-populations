import numpy as np
import matplotlib.pyplot as plt

def cross(x, y, rng):
    rand_bits = rng.binomial(1, 0.5, x.shape)
    return x*rand_bits+y*(1-rand_bits)

def mutate(x, p_m, rng):
    rand_bits = rng.binomial(1, p_m, x.shape)
    return (1-x)*rand_bits+x*(1-rand_bits)

def hamming_distance(x, y):
    return int(np.sum((x-y)**2))


def hamming_profile(P, k, H, unlucky, old, offspring):
    for i in range(P.shape[0]):
        if(i!=unlucky):
            h = hamming_distance(P[i], old)
            H[h] = H[h] - 1
            h = hamming_distance(P[i], offspring)
            H[h] = H[h] + 1
    return H

def run_experiment(mu,n,k,max_count,rng):
    p_m = 1/n
    initial_species = np.ones(n)
    initial_species[rng.choice(n, k, replace=False)] = 0
    P = np.zeros(shape=(mu, n))
    P[:,:] = initial_species
    hamming_p = {2*i:[0] for i in range(k+1)}
    hamming_p[0] = [1]
    count = 0
    run_count = -1
    H = {2*i: 0 for i in range(k+1)}  # 0,2,...,2k
    H[0] = mu*(mu-1)*0.5
    while (count < max_count):
        if(count%1000==0):
            print(count)
        i = rng.integers(mu)
        j = rng.integers(mu)
        offspring = mutate(cross(P[i], P[j], rng), p_m, rng)
        ones = np.sum(offspring)
        if (ones == n):
            # global optimum
            run_count = count
            break
        elif (ones == n-k):
            # on the plateau
            unlucky = rng.integers(mu+1)
            if (unlucky != mu):
                old = P[unlucky].copy()
                P[unlucky] = offspring.copy()
                H = hamming_profile(P, k, H, unlucky, old, offspring)
        for i in range(k+1):
            hamming_p[2*i].append(H[2*i]/(mu*(mu-1)*0.5))
        count += 1
    return count, hamming_p, run_count

if __name__ == '__main__':
    rng = np.random.default_rng(11)
    smallInstance = True
    if smallInstance:
        # Figure 1a
        # number of fitness evaluations
        n = 100
        mu = 20
        k = 5
        max_count = int(6e5)
    else:
        # Figure 1b
        # number of fitness evaluations
        n = 1000
        mu = 200
        k = 5
        max_count = int(6e5)

    count,hamming_p,run_count = run_experiment(mu, n, k, max_count,rng)
    if(run_count!=-1):
        print('global optimum found')
    # hamming distance
    for i in range(k+1):
        plt.plot(range(count+1),hamming_p[2*i],label=f'{2*i}')
    plt.xlabel('iteration')
    plt.ylabel('relative frequency of Hamming distances')
    plt.legend()
    plt.savefig(f'relative_frequency_mu{mu}_n{n}_k{k}.png')
