""" Runner file for FLMDP experiments. """

import itertools as it
import pickle
from math import sqrt

import numpy as np

from flmdp import FLMDP
from policy_approximators import naive_approx, sparsity_corrected_approx
from step_is import step_is
from utils import history_action_tuples


def main():
    """ Main experiment. """
    # Trajectory Params
    time_horizon = 20
    n_samples = 100
    gamma = 0.9
    gamma = 0.9

    rmses_naive = dict()
    rmses_sc = dict()
    perfect_returns = dict()
    naive_returns = dict()
    sc_returns = dict()
    true_returns = dict()

    for history_length, state_size, action_size, sigma, reward, use_approximable_pi in it.product(
            range(1, 2), range(2, 3), range(2, 3), range(0, 1), range(0, 1),
        [False]):
        print(
            f"l={history_length}, mag_S={state_size}, mag_A={action_size}, sigma={sigma}, "
            + f"reward={reward}, use_approximable_pi={use_approximable_pi}")

        # Deterministic initial state distribution
        initial_state_probability = np.zeros((state_size))
        initial_state_probability[0] = 1.0

        transition_probability = FLMDP.random_transition_probability(
            state_size=state_size,
            action_size=action_size,
            history_length=history_length,
            mean_reward=reward)
        lmdp = FLMDP(
            state_size=state_size,
            action_size=action_size,
            transition_probability=transition_probability,
            initial_state_probability=initial_state_probability,
            history_length=history_length)

        if use_approximable_pi:
            pi_b = FLMDP.scips_approximable_pi(
                lmdp=lmdp, gamma=gamma, sigma=sigma)
        else:
            pi_b = FLMDP.random_pi(lmdp=lmdp)
        pi_e = FLMDP.random_pi(lmdp=lmdp)

        s_b, a_b, r_b = lmdp.simulate(
            policy=pi_b, time_horizon=time_horizon, n_samples=n_samples)
        _, _, r_e = lmdp.simulate(
            policy=pi_e, time_horizon=time_horizon, n_samples=n_samples)

        # Naive Monte-Carlo Policy Estimator
        hat_b = naive_approx(
            states=s_b,
            actions=a_b,
            rewards=r_b,
            history_length=history_length)

        # Sparsity Corrected Policy Estimator
        tilde_b = sparsity_corrected_approx(
            states=s_b, actions=a_b, rewards=r_b, gamma=gamma, lmdp=lmdp)

        rho_pi = step_is(
            pi_b=pi_b,
            pi_e=pi_e,
            state_samples=s_b,
            action_samples=a_b,
            reward_samples=r_b,
            history_length=history_length,
            gamma=gamma)
        rho_hat = step_is(
            pi_b=hat_b,
            pi_e=pi_e,
            state_samples=s_b,
            action_samples=a_b,
            reward_samples=r_b,
            history_length=history_length,
            gamma=gamma)
        rho_tilde = step_is(
            pi_b=tilde_b,
            pi_e=pi_e,
            state_samples=s_b,
            action_samples=a_b,
            reward_samples=r_b,
            history_length=history_length,
            gamma=gamma)

        # All the data is there, now to turn it into statistics

        # First we compute the RMSE of the two approximations
        rmse_naive = 0.0
        rmse_sc = 0.0
        for history_action in history_action_tuples(state_size, action_size,
                                                    history_length):
            rmse_naive += (pi_b[history_action] - hat_b.get(
                (history_action), 0))**2
            rmse_sc += (pi_b[history_action] - tilde_b.get(
                (history_action), 0))**2
        rmses_naive[(history_length, state_size, action_size, sigma, reward,
                     use_approximable_pi)] = sqrt(rmse_naive)
        rmses_sc[(history_length, state_size, action_size, sigma, reward,
                  use_approximable_pi)] = sqrt(rmse_sc)

        # Then we record the estimated return from the three methods
        perfect_returns[(history_length, state_size, action_size, sigma,
                         reward, use_approximable_pi)] = rho_pi
        naive_returns[(history_length, state_size, action_size, sigma, reward,
                       use_approximable_pi)] = rho_hat
        sc_returns[(history_length, state_size, action_size, sigma, reward,
                    use_approximable_pi)] = rho_tilde

        # Now let's estimate the true return from our sample.
        total_returns = 0.0
        for episode in range(r_e.shape[0]):
            current_return = 0.0
            for time in range(r_e.shape[1]):
                current_return += (gamma**time) + r_e[episode, time]
            total_returns += current_return
        average_return = total_returns / r_e.shape[0]
        true_returns[(history_length, state_size, action_size, sigma, reward,
                      use_approximable_pi)] = average_return

        pickle.dump(rmses_naive, open('rmses_native.pickle', 'wb'))
        pickle.dump(rmses_sc, open('rmses_sc.pickle', 'wb'))
        pickle.dump(perfect_returns, open('perfect_returns.pickle', 'wb'))
        pickle.dump(naive_returns, open('naive_returns.pickle', 'wb'))
        pickle.dump(sc_returns, open('sc_returns.pickle', 'wb'))
        pickle.dump(true_returns, open('true_returns.pickle', 'wb'))


if __name__ == '__main__':
    main()
