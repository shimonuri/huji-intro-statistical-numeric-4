import dataclasses
import random
import calculations


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
    total_energy_expected_value: float = 0
    ground_level: EnergyLevel = EnergyLevel(0, 0, 0)

    def add(self, zero_energy_occurrences, total_energy):
        self.total_energy_expected_value = (
            self.total_energy_expected_value * self.steps + total_energy
        ) / (self.steps + 1)
        self.steps += 1
        self.ground_level.add(zero_energy_occurrences)

    def copy(self, attempt):
        self.steps = attempt.steps
        self.total_energy_expected_value = attempt.total_energy_expected_value
        self.ground_level.copy(attempt.ground_level)


class Particles:
    def __init__(self, max_energy_level, number_of_particles):
        self.max_energy_level = max_energy_level
        self.number_of_particles = number_of_particles
        self.energy_level_to_occurrences = {
            energy_level: 0 for energy_level in range(max_energy_level + 1)
        }
        # put all particles in the ground state
        self.energy_level_to_occurrences[0] = number_of_particles
        self.energy_level_to_probability = {
            energy_level: self.energy_level_to_occurrences[energy_level]
            / self.number_of_particles
            for energy_level in range(max_energy_level + 1)
        }

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


class Run:
    def __init__(self, temperature, max_energy_level, number_of_particles, mu):
        self.temperature = temperature
        self.particles = Particles(max_energy_level, number_of_particles)
        self.data = RunData(temperature=temperature, mu=mu)

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
        for i in range(1, self.particles.max_energy_level + 1):
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

    def _update_probability(self, energy_level):
        self.particles.energy_level_to_probability[energy_level] = (
            self.particles.energy_level_to_occurrences[energy_level]
            / self.particles.number_of_particles
        )

    def _increase_energy(self, energy_level):
        self.particles.energy_level_to_occurrences[energy_level] -= 1
        self.particles.energy_level_to_occurrences[energy_level + 1] += 1
        self._update_probability(energy_level)
        self._update_probability(energy_level + 1)

    def _decrease_energy(self, energy_level):
        self.particles.energy_level_to_occurrences[energy_level] -= 1
        self.particles.energy_level_to_occurrences[energy_level - 1] += 1
        self._update_probability(energy_level)
        self._update_probability(energy_level - 1)


class Model:
    def __init__(
        self, number_of_particles, max_energy_level, temperature, stop_condition
    ):
        self.number_of_particles = number_of_particles
        self.max_energy_level = max_energy_level
        self.temperature = temperature
        self.stop_condition = stop_condition
        self.mu = calculations.find_mu(
            temperature=temperature, number_of_particles=number_of_particles
        )
        self.run = Run(temperature, max_energy_level, number_of_particles, self.mu)

    def run(self, initial_steps=1000) -> Run:
        steps = initial_steps // 2
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
            half_attempt.copy(full_attempt)
            half_attempt = self._run_attempt(half_attempt, steps // 2)
            full_attempt.copy(half_attempt)
            full_attempt = self._run_attempt(full_attempt, steps // 2)

        return full_attempt

    @staticmethod
    def _run_attempt(attempt, steps) -> Run:
        for _ in range(steps):
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
