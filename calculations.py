import numpy as np
from constants import *


def g(energy_level):
    # Calculates the degeneracy in the system
    n = energy_level
    return np.multiply(np.divide(1, 2), np.power(n, 2) + np.multiply(3, n)) + 1


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
    return 1 - get_decrease_probability(mu, temperature, energy_level)


def get_decrease_probability(mu, temperature, energy_level):
    T, n = temperature, energy_level
    beta = np.divide(1, T)
    if n == 0:
        plus_state = np.divide(g(n + 1), (np.exp(beta * (n + 1 - mu)) - 1))
        minus_state = np.divide(g(n), (np.exp(beta * (n - mu)) - 1))
    else:
        plus_state = np.divide(g(n + 1), (np.exp(beta * (n + 1 - mu)) - 1))
        minus_state = np.divide(g(n - 1), (np.exp(beta * (n - 1 - mu)) - 1))
    return np.divide(minus_state, plus_state + minus_state)


def get_specific_heat_capacities(total_energies, temperatures):
    heat_capacities = []
    for i in range(len(total_energies) - 1):
        heat_capacities.append(
            (total_energies[i + 1] - total_energies[i])
            / (temperatures[i + 1] - temperatures[i])
        )
    return heat_capacities


def get_critical_temperature(
    temperature_range, ground_state_expected_values, number_of_particles
):
    x = temperature_range
    y = [
        expected_value / number_of_particles
        for expected_value in ground_state_expected_values
    ]
    for index, temperature in enumerate(x):
        if y[index] < 0.1:
            return temperature
