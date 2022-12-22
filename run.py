import model
from matplotlib import pyplot as plt



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

if __name__ == "__main__":
    #main(number_of_particles=100, temperature=15, stop_condition=1e-1)
