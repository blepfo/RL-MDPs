import itertools as it
import pickle
from math import sqrt

import numpy as np

from flmdp import FLMDP
from policy_approximators import naive_approx, sparsity_corrected_approx
from step_is import step_is

# L-MDP params
base_mag_S = 5
base_mag_A = 5
base_l = 4
base_sigma = 0


# Trajectory Params
T = 20
m = 100
Gamma = 0.9
gamma = 0.9

rmses_naive = list()
rmses_sc = list()
perfect_returns = list()
naive_returns = list()
sc_returns = list()
true_returns = list()

for iteration in range(100):
    l = base_l
    sigma = base_sigma
    mag_A = base_mag_A
    mag_S = base_mag_S
    print(iteration)

    # Deterministic initial state distribution
    P0 = np.zeros((mag_S))
    P0[0] = 1.0

    P = FLMDP.random_P(mag_S=mag_S,
                       mag_A=mag_A,
                       l=l,
                       mean_reward=100)
    lmdp = FLMDP(mag_S=mag_S,
                 mag_A=mag_A,
                 P=P,
                 P0=P0,
                 l=l)

    # pi_b = FLMDP.scips_approximable_pi(lmdp=lmdp,
    #                                    Gamma=Gamma,
    #                                    sigma=sigma)
    pi_b = FLMDP.random_pi(lmdp=lmdp)
    pi_e = FLMDP.random_pi(lmdp=lmdp)

    s_b, a_b, r_b = lmdp.simulate(pi=pi_b,
                                  T=T,
                                  m=m)
    s_e, a_e, r_e = lmdp.simulate(pi=pi_e,
                                  T=T,
                                  m=m)

    # Naive Monte-Carlo Policy Estimator
    hat_b = naive_approx(states=s_b,
                         actions=a_b,
                         rewards=r_b,
                         l=l)

    # Sparsity Corrected Policy Estimator
    tilde_b = sparsity_corrected_approx(states=s_b,
                                        actions=a_b,
                                        rewards=r_b,
                                        Gamma=Gamma,
                                        lmdp=lmdp)

    rho_pi = step_is(pi_b=pi_b,
                     pi_e=pi_e,
                     state_samples=s_b,
                     action_samples=a_b,
                     reward_samples=r_b,
                     l=l,
                     gamma=gamma)
    rho_hat = step_is(pi_b=hat_b,
                      pi_e=pi_e,
                      state_samples=s_b,
                      action_samples=a_b,
                      reward_samples=r_b,
                      l=l,
                      gamma=gamma)
    rho_tilde = step_is(pi_b=tilde_b,
                        pi_e=pi_e,
                        state_samples=s_b,
                        action_samples=a_b,
                        reward_samples=r_b,
                        l=l,
                        gamma=gamma)

    # All the data is there, now to turn it into statistics

    # First we compute the RMSE of the two approximations
    history_sizes = it.repeat(list(range(mag_S+1)), l)
    dist_sizes = it.chain(history_sizes, [list(range(mag_A))])
    history_actions = it.product(*dist_sizes)

    rmse_naive = 0.0
    rmse_sc = 0.0
    for history_action in history_actions:
        rmse_naive += (pi_b[history_action] -
                       hat_b.get((history_action), 0))**2
        rmse_sc += (pi_b[history_action] - tilde_b.get((history_action), 0))**2
    rmses_naive.append(sqrt(rmse_naive))
    rmses_sc.append(sqrt(rmse_sc))

    # Then we record the estimated return from the three methods
    perfect_returns.append(rho_pi)
    naive_returns.append(rho_hat)
    sc_returns.append(rho_tilde)

    # Now let's estimate the true return from our sample.
    total_returns = 0.0
    for episode in range(r_e.shape[0]):
        r = 0.0
        for time in range(r_e.shape[1]):
            r += (gamma ** time) + r_e[episode, time]
        total_returns += r
    average_return = total_returns / r_e.shape[0]
    true_returns.append(average_return)

pickle.dump(rmses_naive, open('rmses_native_random.pickle', 'wb'))
pickle.dump(rmses_sc, open('rmses_sc_random.pickle', 'wb'))
pickle.dump(perfect_returns, open('perfect_returns_random.pickle', 'wb'))
pickle.dump(naive_returns, open('naive_returns_random.pickle', 'wb'))
pickle.dump(sc_returns, open('sc_returns_random.pickle', 'wb'))
pickle.dump(true_returns, open('true_returns_random.pickle', 'wb'))
