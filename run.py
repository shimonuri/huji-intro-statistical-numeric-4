import model


def main(number_of_particles, temperature, stop_condition):
    my_model = model.Model(
        number_of_particles=number_of_particles,
        temperature=temperature,
        stop_condition=stop_condition,
    )
    result = my_model.run()
    print(result)


if __name__ == "__main__":
    main(number_of_particles=100, temperature=0.2, stop_condition=1e-3)
