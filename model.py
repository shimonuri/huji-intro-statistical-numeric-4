import dataclasses
import random
import calculations
import constants
import logging
import numpy as np


@dataclasses.dataclass
class EnergyLevel:
    level: int
    sum: float = 0
    square_sum: float = 0
    add_count: int = 0

    @property
    def expected_value(self):
        return self.sum / self.add_count

    @property
    def second_momentum(self):
        return self.square_sum / self.add_count

    @property
    def variance(self):
        return self.second_momentum - self.expected_value ** 2

    @property
    def std(self):
        return self.variance ** 0.5

    def add(self, occurrences):
        self.sum += occurrences
        self.square_sum += occurrences ** 2
        self.add_count += 1

    def copy(self, energy_level):
        self.level = energy_level.level
        self.sum = energy_level.sum
        self.square_sum = energy_level.square_sum
        self.add_count = energy_level.add_count


@dataclasses.dataclass
class RunData:
    temperature: float
    mu: float
    ground_level: EnergyLevel
    steps: int = 0
    total_energy_sum: int = 0
    total_energy_square_sum: int = 0

    @property
    def total_energy_expected_value(self):
        return self.total_energy_sum / self.steps

    @property
    def total_energy_second_momentum(self):
        return self.total_energy_square_sum / self.steps

    @property
    def total_energy_std(self):
        return (self.total_energy_second_momentum - self.total_energy_expected_value ** 2) ** 0.5

    def add(self, ground_state_occurrences, total_energy):
        self.total_energy_sum += total_energy
        self.total_energy_square_sum += total_energy ** 2
        self.steps += 1
        self.ground_level.add(ground_state_occurrences)

    def copy(self, attempt):
        self.steps = attempt.steps
        self.total_energy_sum = attempt.total_energy_sum
        self.total_energy_square_sum = attempt.total_energy_square_sum
        self.ground_level.copy(attempt.ground_level)


class Particles:
    def __init__(self, max_energy_level, number_of_particles):
        self.max_energy_level = max_energy_level
        self.number_of_particles = int(number_of_particles)
        self._set_initial_condition(max_energy_level, self.number_of_particles)

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
        self.data = RunData(
            temperature=temperature, mu=mu, ground_level=EnergyLevel(level=0)
        )
        self.energy_level_to_decrease_probability = {
            energy_level: calculations.get_decrease_probability(
                mu=mu, temperature=self.temperature, energy_level=energy_level
            )
            for energy_level in range(max_energy_level + 1)
        }

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
        energy_level = random.choices(
            *zip(*self.particles.energy_level_to_occurrences.items())
        )
        return energy_level[0]

    def _update_energy(self, energy_level):
        random_number = random.random()
        if random_number <= self.energy_level_to_decrease_probability[energy_level]:
            self._decrease_energy(energy_level)
        else:
            self._increase_energy(energy_level)

    def _increase_energy(self, energy_level):
        if energy_level == self.particles.max_energy_level:
            return
        self.particles.energy_level_to_occurrences[energy_level] -= 1
        self.particles.energy_level_to_occurrences[energy_level + 1] += 1

    def _decrease_energy(self, energy_level):
        if energy_level == 0:
            return
        self.particles.energy_level_to_occurrences[energy_level] -= 1
        self.particles.energy_level_to_occurrences[energy_level - 1] += 1


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
        steps = int(self.number_of_particles * 1e2 // 2)
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
            if steps > constants.MAX_STEPS:
                logging.info(f"Max steps reached: {steps}")
                return full_attempt
            logging.info(f"Running {steps:.0e} steps")
            half_attempt.copy(full_attempt)
            half_attempt = self._run_attempt(half_attempt, steps // 2)
            full_attempt.copy(half_attempt)
            full_attempt = self._run_attempt(full_attempt, steps // 2)

        return full_attempt

    def _run_attempt(self, attempt, steps) -> Run:
        for i in range(steps):
            if i % int(round(2 * steps // 5)) == 0:
                logging.info(
                    f"Currently in Step: {i:.1e} / {steps:.1e}, (Temperature={self.temperature})"
                )

            attempt.run_step()

        return attempt

    def _should_stop(self, half_attempt, full_attempt) -> bool:
        if half_attempt.data.steps == 0 or full_attempt.data.steps == 0:
            return False

        return (
            np.divide(
                np.abs(
                    full_attempt.data.ground_level.expected_value
                    - half_attempt.data.ground_level.expected_value
                ),
                full_attempt.data.ground_level.expected_value,
            )
            <= self.stop_condition
        )
