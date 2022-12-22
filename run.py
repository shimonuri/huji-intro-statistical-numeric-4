import model
from matplotlib import pyplot as plt
import numpy as np


# update matplotlib params:
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.size": 16})


def run_multiple_models(number_of_particles):
    temperatures = _get_temperatures(number_of_particles)
    ground_state_expected_values, ground_state_stds = multiple_temperature_runs(
        number_of_particles, temperatures
    )
    plot_ground_state_expected_value(
        temperature_range=temperatures,
        ground_state_expected_value=ground_state_expected_values,
        number_of_particles=number_of_particles,
        ground_state_std_lst=ground_state_stds,
    )


def multiple_temperature_runs(number_of_particles, temperatures):
    ground_state_expected_values = []
    ground_state_stds = []
    for temperature in temperatures:
        current_model = model.Model(
            number_of_particles=number_of_particles,
            temperature=temperature,
            stop_condition=_get_stop_condition(temperature),
        )
        result = current_model.run()
        ground_state_expected_values.append(result.data.ground_level.expected_value)
        ground_state_stds.append(result.data.ground_level.std)
    return ground_state_expected_values, ground_state_stds


def _get_temperatures(number_of_particles):
    max_temperature = _get_max_temperature(number_of_particles)
    return [
        0.2 * temperature for temperature in range(1, int(max_temperature // 0.2) + 1)
    ]


def _get_max_temperature(number_of_particles):
    if number_of_particles == 1e4:
        return 25

    return round(5 * np.log10(number_of_particles))


def _get_stop_condition(temperature):
    if temperature <= 1:
        return 1e-3
    elif temperature <= 2:
        return 5e-3
    else:
        return 1e-2


def plot_ground_state_expected_value(
    temperature_range,
    ground_state_expected_value,
    number_of_particles,
    ground_state_std_lst,
):
    x = temperature_range
    y = [
        expected_value / number_of_particles
        for expected_value in ground_state_expected_value
    ]
    plt.plot(x, y)
    plt.xlabel(r"$T$")
    plt.ylabel(r"$\left\langle N_{0}\right\rangle /N$")
    plt.legend()



if __name__ == "__main__":
    run_multiple_models(number_of_particles=100)
