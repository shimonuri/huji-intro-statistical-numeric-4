import model
from matplotlib import pyplot as plt
import numpy as np
import calculations
import click
import logging
import json
import multiprocessing
import signal
import os
import pathlib

logging.getLogger().setLevel(logging.INFO)

# update matplotlib params:
# plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.size": 16})


@click.command()
@click.argument("path", type=click.Path(exists=False))
@click.option("--particles", type=int, default=None)
@click.option("--plot", is_flag=True, default=False)
@click.option("--fast", is_flag=True, default=False)
@click.option("--processes", "-p", type=int, default=1)
def main(path, particles, plot, fast, processes):
    if not plot:
        if particles is None:
            run_multiple_models(path, fast=fast, processes=processes)
        else:
            run_multiple_models(
                path, numbers_of_particles=[particles], fast=fast, processes=processes
            )
    else:
        with open(path, "rt") as file:
            data = json.load(file)
        plot_data(data)


def plot_data(data):
    _plot_ground_states_with_stds(data)
    _plot_ground_states_with_no_stds(data)
    plot_critical_temperature(data)
    _plot_heat_capacities(data)


def _plot_heat_capacities(data):
    if len(data) == 4:
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        fig.tight_layout(pad=3.0)
    else:
        axs = None
    for i, (number_of_particles, number_of_particles_data) in enumerate(data.items()):
        critical_temperature = calculations.get_critical_temperature(
            temperature_range=number_of_particles_data["temperatures"],
            ground_state_expected_values=number_of_particles_data[
                "ground_state_expected_values"
            ],
            number_of_particles=int(number_of_particles),
        )
        plot_specific_heat_capacity(
            total_energies=number_of_particles_data["total_energy_expected_values"],
            temperatures=number_of_particles_data["temperatures"],
            number_of_particles=int(number_of_particles),
            critical_temperature=critical_temperature,
            ax=None if axs is None else axs.flatten()[i],
        )
    plt.show()


def _plot_ground_states_with_no_stds(data):
    if len(data) == 4:
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        fig.tight_layout(pad=3.0)
    else:
        axs = None

    for i, (number_of_particles, number_of_particles_data) in enumerate(data.items()):
        # create 4 subplots
        plot_ground_state_expected_value(
            temperature_range=number_of_particles_data["temperatures"],
            ground_state_expected_values=number_of_particles_data[
                "ground_state_expected_values"
            ],
            ground_state_stds=number_of_particles_data["ground_state_stds"],
            number_of_particles=int(float(number_of_particles)),
            ax=None if axs is None else axs.flatten()[i],
            add_errorbar=False,
        )
    plt.show()


def _plot_ground_states_with_stds(data):
    if len(data) == 4:
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        fig.tight_layout(pad=3.0)
    else:
        axs = None

    for i, (number_of_particles, number_of_particles_data) in enumerate(data.items()):
        # create 4 subplots
        plot_ground_state_expected_value(
            temperature_range=number_of_particles_data["temperatures"],
            ground_state_expected_values=number_of_particles_data[
                "ground_state_expected_values"
            ],
            ground_state_stds=number_of_particles_data["ground_state_stds"],
            number_of_particles=int(float(number_of_particles)),
            ax=None if axs is None else axs.flatten()[i],
            add_errorbar=True,
        )
    plt.show()


def run_multiple_models(path, numbers_of_particles=None, fast=False, processes=1):
    if numbers_of_particles is None:
        numbers_of_particles = [1e1, 1e2, 1e3, 1e4]

    number_of_particles_to_data = {}
    for number_of_particles in numbers_of_particles:
        logging.info(f"Number of particles: {number_of_particles}")
        if fast:
            temperatures = [0.2, 1]
        else:
            temperatures = _get_temperatures(number_of_particles, step_side=0.2)
        (
            ground_state_expected_values,
            ground_state_stds,
            total_energy,
            total_energy_stds,
        ) = multiple_temperature_runs(
            number_of_particles, temperatures, processes, path
        )
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
            "total_energy_expected_values": total_energy,
            "total_energy_stds": total_energy_stds,
        }
        with open(path, "wt") as file:
            json.dump(
                number_of_particles_to_data, file, indent=4,
            )


def _initialize_process(parent_pid):
    def _handle_sigint():
        os.kill(parent_pid, signal.SIGINT)

    signal.signal(signal.SIGINT, _handle_sigint)


def multiple_temperature_runs(number_of_particles, temperatures, processes, path):
    ground_state_expected_values = []
    ground_state_stds = []
    total_energy = []
    total_energy_stds = []
    try:
        with multiprocessing.Pool(
            processes=processes, initializer=_initialize_process, initargs=[os.getpid()]
        ) as pool:
            results = pool.starmap(
                _run_model,
                [
                    (
                        number_of_particles,
                        temperature,
                        pathlib.Path(path).with_suffix(f".{temperature}.json"),
                    )
                    for temperature in temperatures
                ],
            )
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, terminating...")
        pool.terminate()
        pool.join()
        raise

    for result in results:
        ground_state_expected_values.append(result.data.ground_level.expected_value)
        ground_state_stds.append(result.data.ground_level.std)
        total_energy_stds.append(result.data.total_energy_std)
        total_energy.append(result.data.total_energy_expected_value)
    return (
        ground_state_expected_values,
        ground_state_stds,
        total_energy,
        total_energy_stds,
    )


def _run_model(number_of_particles, temperature, path):
    current_model = model.Model(
        number_of_particles=number_of_particles,
        temperature=temperature,
        stop_condition=_get_stop_condition(temperature),
    )
    result = current_model.run()
    with open(path.as_posix(), "wt") as file:
        json.dump(
            {
                "number_of_particles": number_of_particles,
                "temperature": temperature,
                "ground_state_expected_value": result.data.ground_level.expected_value,
                "ground_state_std": result.data.ground_level.std,
                "total_energy_expected_value": result.data.total_energy_expected_value,
                "total_energy_std": result.data.total_energy_std,
            },
            file,
            indent=4,
        )
    return result


def _get_temperatures(number_of_particles, step_side=0.2):
    max_temperature = _get_max_temperature(number_of_particles)
    return [
        step_side * temperature
        for temperature in range(1, int(max_temperature // step_side) + 2)
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
    ground_state_expected_values,
    number_of_particles,
    ground_state_stds,
    add_errorbar=True,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    x = temperature_range
    y = [
        expected_value / number_of_particles
        for expected_value in ground_state_expected_values
    ]
    ax.plot(x, y, label=f"number of particles: {number_of_particles}")
    if add_errorbar:
        ax.errorbar(
            x,
            y,
            yerr=[
                std / expected_value
                for expected_value, std in zip(
                    ground_state_expected_values, ground_state_stds
                )
            ],
            fmt="none",
            ecolor="black",
            elinewidth=0.5,
            capsize=2,
        )
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Ground state expected value")
    ax.set_title(f"Number of particles: {number_of_particles}")


def plot_specific_heat_capacity(
    number_of_particles,
    temperatures,
    total_energies,
    critical_temperature=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    heat_capacities = calculations.get_specific_heat_capacities(
        total_energies=total_energies, temperatures=temperatures,
    )
    if critical_temperature is not None:
        ax.axvline(x=critical_temperature, color="red", linestyle="--")

    ax.plot(temperatures[:-1], heat_capacities, "bo")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Specific heat capacity")
    ax.set_title(f"Number of particles: {number_of_particles}")

def plot_critical_temperature(data):
    critical_temperatures = []
    for number_of_particles, number_of_particles_data in data.items():
        critical_temperatures.append(
            calculations.get_critical_temperature(
                temperature_range=number_of_particles_data["temperatures"],
                ground_state_expected_values=number_of_particles_data[
                    "ground_state_expected_values"
                ],
                number_of_particles=int(number_of_particles),
            )
        )
    print(critical_temperatures)
    plt.plot([int(key) for key in data.keys()], critical_temperatures)
    plt.xlabel("Number of particles")
    plt.ylabel("Critical temperature")
    plt.loglog()
    plt.grid()
    plt.title("Critical temperature (log-log scale)")
    plt.show()


if __name__ == "__main__":
    main()
