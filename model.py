import dataclasses


@dataclasses.dataclass
class EnergyLevel:
    level: int
    expected_value: float
    second_momentum: float
    add_count: int = 0

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
class Attempt:
    steps: int = 0
    total_energy_expected_value: float = 0
    ground_energy_level: EnergyLevel = EnergyLevel(0, 0, 0)

    def add(self, zero_energy_occurrences, total_energy):
        self.total_energy_expected_value = (
            self.total_energy_expected_value * self.steps + total_energy
        ) / (self.steps + 1)
        self.steps += 1
        self.ground_energy_level.add(zero_energy_occurrences)

    def copy(self, attempt):
        self.steps = attempt.steps
        self.total_energy_expected_value = attempt.total_energy_expected_value
        self.ground_energy_level.copy(attempt.ground_energy_level)


class Particles:
    def __init__(self, max_energy_level, number_of_particles):
        self.max_energy_level = max_energy_level
        self.number_of_particles = number_of_particles
        self.energy_level_to_occurrences = {
            energy_level: 0 for energy_level in range(max_energy_level + 1)
        }
        # put all particles in the ground state
        self.energy_level_to_occurrences[0] = number_of_particles

    @property
    def energy(self):
        return sum(
            energy_level * occurrences
            for energy_level, occurrences in self.energy_level_to_occurrences.items()
        )


class Model:
    def __init__(self, temperature, stop_condition):
        self.temperature = temperature
        self.stop_condition = stop_condition

    def run(self, initial_steps=1000):
        steps = initial_steps
        half_attempt = Attempt()
        full_attempt = Attempt()
        while not self._should_stop(half_attempt, full_attempt):
            half_attempt = self._run_attempt(half_attempt, steps // 2)
            full_attempt.copy(half_attempt)
            full_attempt = self._run_attempt(full_attempt, steps // 2)

    def _run_attempt(self, attempt, steps) -> Attempt:
        for _ in range(steps):
            zero_energy_occurrences, total_energy = self._run_step()
            attempt.add(zero_energy_occurrences, total_energy)
        return attempt

    def _should_stop(self, half_attempt, full_attempt) -> bool:
        if half_attempt.steps == 0 or full_attempt.steps == 0:
            return False

        return (
            abs(
                full_attempt.ground_energy_level.expected_value
                - half_attempt.ground_energy_level.expected_value
            )
            / full_attempt.ground_energy_level.expected_value
            <= self.stop_condition
        )
