import model
from matplotlib import pyplot as plt


# update matplotlib params:
plt.rcParams.update({'usetex': True})
plt.rcParams.update({'font.size': 16})

def main(number_of_particles, temperature, stop_condition):
    my_model = model.Model(
        number_of_particles=number_of_particles,
        temperature=temperature,
        stop_condition=stop_condition,
    )
    result = my_model.run()
    plt.plot(
        list(result.particles.energy_level_to_occurrences.keys()),
        list(result.particles.energy_level_to_occurrences.values()),
    )
    plt.show()

def multiple_temperature_runs(number_of_particles, temperatures, stop_condition):
    for temperature in temperatures:
        main(number_of_particles, temperature, stop_condition)

def plot_ground_state_expected_value(temperature_range,
                                     ground_state_expected_value,
                                     number_of_particles,
                                     ground_state_std_lst):
    x = temperature_range
    y = [expected_value / number_of_particles for expected_value in ground_state_expected_value]
    plt.plot(x, y)
    plt.xlabel(r'$T$')
    plt.ylabel(r'$\left\langle N_{0}\right\rangle /N$')


if __name__ == "__main__":
    #main(number_of_particles=100, temperature=15, stop_condition=1e-1)
