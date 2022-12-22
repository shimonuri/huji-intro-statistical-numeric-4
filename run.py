import model


def main(max_energy_level, number_of_particles, temperature, stop_condition):
    my_model = model.Model(
        number_of_particles=number_of_particles,
        max_energy_level=max_energy_level,
        temperature=temperature,
        stop_condition=stop_condition,
    )
    result = my_model.run()


if __name__ == "__main__":
    main(
        max_energy_level=100, number_of_particles=100, temperature=1, stop_condition=100
    )
