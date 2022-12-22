import model


def main(temperature, stop_condition):
    my_model = model.Model(temperature, stop_condition)
    result = my_model.run()


if __name__ == "__main__":
    main(1, 0.2)
