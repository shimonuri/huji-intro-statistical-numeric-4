import numpy as np
from constants import *



def g(energy_level):
    # Calculates the degeneracy in the system
    n = energy_level
    return 1 / 2 * n * (n + 3) + 1


def get_number_of_particles(mu, temperature):
    T = temperature
    beta = 1 / T
    return sum(
        [g(n) / (np.exp(beta * (n - mu)) - 1) for n in range(MAX_ENERGY_LEVEL + 1)]
    )


def find_mu(temperature, number_of_particles):
    T, N = temperature, number_of_particles
    mu_min = MU_MIN
    mu_max = 0

    mu_try = (mu_min + mu_max) / 2
    N_try = get_number_of_particles(mu=mu_try, temperature=T)

    while N != int(N_try):
        mu_try = (mu_min + mu_max) / 2
        N_try = get_number_of_particles(mu=mu_try, temperature=T)
        if N_try > N:
            mu_max = mu_try
        else:
            mu_min = mu_try

    return mu_try


def get_increase_probability(mu, temperature, energy_level):
    T, n = temperature, energy_level
    beta = 1 / T
    plus_state = g(n + 1) / (np.exp(beta * (n + 1 - mu)) - 1)
    minus_state = g(n - 1) / (np.exp(beta * (n - 1 - mu)) - 1)
    return plus_state / (plus_state + minus_state)


def get_decrease_probability(mu, temperature, energy_level):
    T, n = temperature, energy_level
    beta = 1 / T
    plus_state = g(n + 1) / (np.exp(beta * (n + 1 - mu)) - 1)
    minus_state = g(n - 1) / (np.exp(beta * (n - 1 - mu)) - 1)
    return minus_state / (plus_state + minus_state)

