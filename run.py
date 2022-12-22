import model
import matplotlib.pyplot as plt

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
    print(result)


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
    main(number_of_particles=100, temperature=0.2, stop_condition=1e-3)
