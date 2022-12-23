import dataclasses
import random
import calculations
import constants
import logging


@dataclasses.dataclass
class EnergyLevel:
    level: int
    expected_value: float
    second_momentum: float
    add_count: int = 0

    @property
    def variance(self):
        return self.second_momentum - self.expected_value ** 2

    @property
    def std(self):
        return self.variance ** 0.5

    def add(self, occurrences):
        self.expected_value = (self.expected_value * self.add_count + occurrences) / (
            self.add_count + 1
        )
        self.second_momentum = (
            self.second_momentum * self.add_count + occurrences ** 2
        ) / (self.add_count + 1)
        self.add_count += 1

    def copy(self, energy_level):
        self.level = energy_level.level
        self.expected_value = energy_level.expected_value
        self.second_momentum = energy_level.second_momentum
        self.add_count = energy_level.add_count


@dataclasses.dataclass
class RunData:
    temperature: float
    mu: float
    steps: int = 0
    total_energy_second_momentum: float = 0
    total_energy_expected_value: float = 0
    ground_level: EnergyLevel = EnergyLevel(0, 0, 0)

    @property
    def total_energy_std(self):
        return (
            self.total_energy_second_momentum - self.total_energy_expected_value ** 2
        ) ** 0.5

    def add(self, zero_energy_occurrences, total_energy):
        self.total_energy_expected_value = (
            self.total_energy_expected_value * self.steps + total_energy
        ) / (self.steps + 1)
        self.total_energy_second_momentum = (
            self.total_energy_second_momentum * self.steps + total_energy ** 2
        ) / (self.steps + 1)
        self.steps += 1
        self.ground_level.add(zero_energy_occurrences)

    def copy(self, attempt):
        self.steps = attempt.steps
        self.total_energy_expected_value = attempt.total_energy_expected_value
        self.ground_level.copy(attempt.ground_level)
        self.total_energy_second_momentum = attempt.total_energy_second_momentum


class Particles:
    def __init__(self, max_energy_level, number_of_particles):
        self.max_energy_level = max_energy_level
        self.number_of_particles = number_of_particles
        self._set_initial_condition(max_energy_level, number_of_particles)
        self.energy_level_to_probability = {
            energy_level: self._get_energy_level_probability(energy_level)
            for energy_level in range(max_energy_level + 1)
        }
        self.energy_level_to_probability[-1] = 0

    def __str__(self):
        return str(self.energy_level_to_occurrences)

    def copy(self, particles):
        self.max_energy_level = particles.max_energy_level
        self.number_of_particles = particles.number_of_particles
        for energy_level, occurrences in particles.energy_level_to_occurrences.items():
            self.energy_level_to_occurrences[energy_level] = occurrences

    @property
    def energy(self):
        return sum(
            energy_level * occurrences
            for energy_level, occurrences in self.energy_level_to_occurrences.items()
        )

    def update_probability(self, energy_level):
        for el in range(energy_level + 1):
            self.energy_level_to_probability[el] = self._get_energy_level_probability(
                el
            )

    def _get_energy_level_probability(self, energy_level):
        return sum(
            self.energy_level_to_occurrences[el] / self.number_of_particles
            for el in range(energy_level + 1)
        )

    def _set_initial_condition(self, max_energy_level, number_of_particles):
        self.energy_level_to_occurrences = {
            energy_level: 0 for energy_level in range(max_energy_level + 1)
        }
        for _ in range(number_of_particles):
            self.energy_level_to_occurrences[random.randint(0, max_energy_level)] += 1


class Run:
    def __init__(self, temperature, max_energy_level, number_of_particles, mu):
        self.temperature = temperature
        self.particles = Particles(max_energy_level, number_of_particles)
        self.data = RunData(temperature=temperature, mu=mu)

    def __str__(self):
        return str(self.particles)

    def run_step(self):
        energy_level = self._get_random_energy_level()
        self._update_energy(energy_level)
        self.data.add(
            self.particles.energy_level_to_occurrences[0], self.particles.energy
        )

    def copy(self, run):
        self.temperature = run.temperature
        self.particles.copy(run.particles)
        self.data.copy(run.data)

    def _get_random_energy_level(self):
        random_number = random.random()
        for i in range(0, self.particles.max_energy_level + 1):
            if (
                self.particles.energy_level_to_probability[i - 1]
                < random_number
                <= self.particles.energy_level_to_probability[i]
            ):
                return i

        raise RuntimeError("Could not determine energy level")

    def _update_energy(self, energy_level):
        random_number = random.random()
        if random_number <= calculations.get_decrease_probability(
            mu=self.data.mu,
            temperature=self.data.temperature,
            energy_level=energy_level,
        ):
            self._decrease_energy(energy_level)
        else:
            self._increase_energy(energy_level)

    def _increase_energy(self, energy_level):
        self.particles.energy_level_to_occurrences[energy_level] -= 1
        self.particles.energy_level_to_occurrences[energy_level + 1] += 1
        self.particles.update_probability(energy_level + 1)

    def _decrease_energy(self, energy_level):
        self.particles.energy_level_to_occurrences[energy_level] -= 1
        self.particles.energy_level_to_occurrences[energy_level - 1] += 1
        self.particles.update_probability(energy_level)


class Model:
    def __init__(self, number_of_particles, temperature, stop_condition):
        self.number_of_particles = number_of_particles
        self.max_energy_level = constants.MAX_ENERGY_LEVEL
        self.temperature = temperature
        self.stop_condition = stop_condition
        self.mu = calculations.find_mu(
            temperature=temperature, number_of_particles=number_of_particles
        )
        logging.info(f"mu: {self.mu}")

    def run(self) -> Run:
        steps = int(self.number_of_particles * 1e3 // 2)
        half_attempt = Run(
            temperature=self.temperature,
            max_energy_level=self.max_energy_level,
            number_of_particles=self.number_of_particles,
            mu=self.mu,
        )
        full_attempt = Run(
            temperature=self.temperature,
            max_energy_level=self.max_energy_level,
            number_of_particles=self.number_of_particles,
            mu=self.mu,
        )
        while not self._should_stop(half_attempt, full_attempt):
            steps *= 2
            logging.info(f"Running {steps:.0e} steps")
            half_attempt.copy(full_attempt)
            half_attempt = self._run_attempt(half_attempt, steps // 2)
            full_attempt.copy(half_attempt)
            full_attempt = self._run_attempt(full_attempt, steps // 2)

        return full_attempt

    @staticmethod
    def _run_attempt(attempt, steps) -> Run:
        for i in range(steps):
            if i % int(round(2 * steps // 100)) == 0:
                logging.info(f"Currently in Step: {i:.1e}")

            attempt.run_step()

        return attempt

    def _should_stop(self, half_attempt, full_attempt) -> bool:
        if half_attempt.data.steps == 0 or full_attempt.data.steps == 0:
            return False

        return (
            abs(
                full_attempt.data.ground_level.expected_value
                - half_attempt.data.ground_level.expected_value
            )
            / full_attempt.data.ground_level.expected_value
            <= self.stop_condition
        )
