import model
from matplotlib import pyplot as plt
import numpy as np
import calculations
import click
import logging
import json

logging.getLogger().setLevel(logging.INFO)

# update matplotlib params:
# plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.size": 16})


@click.command()
@click.argument("path", type=click.Path(exists=False))
@click.option("--plot", is_flag=True, default=False)
def main(path, plot):
    if not plot:
        run_multiple_models(path)
    else:
        with open(path, "rt") as file:
            data = json.load(file)
            for number_of_particles, number_of_particles_data in data:
                plot_ground_state_expected_value(
                    temperature_range=number_of_particles_data["temperatures"],
                    ground_state_expected_values=number_of_particles_data[
                        "ground_state_expected_values"
                    ],
                    ground_state_stds=number_of_particles_data["ground_state_stds"],
                    number_of_particles=number_of_particles,
                )
            plt.show()


def run_multiple_models(path):
    number_of_particles_to_data = {}
    for number_of_particles in [1e1]:  # [1e1, 1e2, 1e3, 1e4]:
        temperatures = _get_temperatures(number_of_particles)
        (
            ground_state_expected_values,
            ground_state_stds,
            total_energy_stds,
        ) = multiple_temperature_runs(number_of_particles, temperatures)
        plot_ground_state_expected_value(
            temperature_range=temperatures,
            ground_state_expected_values=ground_state_expected_values,
            number_of_particles=number_of_particles,
            ground_state_stds=ground_state_stds,
        )
        number_of_particles_to_data[number_of_particles] = {
            "temperatures": temperatures,
            "ground_state_expected_values": ground_state_expected_values,
            "ground_state_stds": ground_state_stds,
            "total_energy_stds": total_energy_stds,
        }
    plt.show()
    with open(path, "wt") as file:
        json.dump(
            number_of_particles_to_data, file, indent=4,
        )


def multiple_temperature_runs(number_of_particles, temperatures):
    ground_state_expected_values = []
    ground_state_stds = []
    total_energy_stds = []
    for temperature in temperatures:
        logging.info(f"Temperature: {temperature}")
        current_model = model.Model(
            number_of_particles=number_of_particles,
            temperature=temperature,
            stop_condition=_get_stop_condition(temperature),
        )
        result = current_model.run()
        ground_state_expected_values.append(result.data.ground_level.expected_value)
        ground_state_stds.append(result.data.ground_level.std)
        total_energy_stds.append(result.data.total_energy_std)
    return ground_state_expected_values, ground_state_stds, total_energy_stds


def _get_temperatures(number_of_particles):
    return [0.2, 5]
    max_temperature = _get_max_temperature(number_of_particles)
    return [
        0.2 * temperature for temperature in range(1, int(max_temperature // 0.2) + 1)
    ]


def _get_max_temperature(number_of_particles):
    if number_of_particles == 1e4:
        return 50

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
    ground_state_expected_values,
    number_of_particles,
    ground_state_stds,
):
    x = temperature_range
    y = [
        expected_value / number_of_particles
        for expected_value in ground_state_expected_values
    ]
    plt.plot(x, y, label=f"number of particles: {number_of_particles}")
    # plt.errorbar(
    #     x,
    #     y,
    #     yerr=[
    #         std / expected_value
    #         for expected_value, std in zip(
    #             ground_state_expected_values, ground_state_stds
    #         )
    #     ],
    #     fmt="none",
    #     ecolor="black",
    #     elinewidth=0.5,
    #     capsize=2,
    # )
    plt.xlabel(r"$T$")
    plt.ylabel(r"$\left\langle N_{0}\right\rangle /N$")
    plt.legend()


<<<<<<< HEAD
def plot_specific_heat_capacity(temperature_range, number_of_particles, total_energy_std):

    cv_list = [calculations.get_specific_heat_capacity(temperature=temperature,
                                                       number_of_particles=number_of_particles,
                                                       total_energy_std=total_energy_std)
               for temperature in temperature_range]
    plt.plot(temperature_range, cv_list)
    plt.xlabel(r"$T$")
    plt.ylabel(r"$c_{v}\left(T\right)$")
    plt.legend()


=======
>>>>>>> 1ca803b9af880ef5d10c5c31b0ffee27b8be7f67
if __name__ == "__main__":
    main()
