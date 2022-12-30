import pathlib

import click
import click_pathlib
import json
import numpy as np


@click.command()
@click.argument("in_path", type=click_pathlib.Path(exists=True))
@click.argument("out_path", type=click_pathlib.Path(exists=False))
def main(in_path: pathlib.Path, out_path: pathlib.Path):
    number_of_particles_to_data = {}
    for file in in_path.iterdir():
        with file.open("rt") as fd:
            old_data = json.load(fd)
        print(file.as_posix())
        for expected_value, second_moment in zip(
                old_data["N_0_avgs"], old_data["N_0_2_avgs"],
        ):
            try:
                np.sqrt(float(second_moment) - np.square(float(expected_value)))
            except Exception:
                import ipdb; ipdb.set_trace()
        number_of_particles_to_data[int(file.stem)] = {
            "temperatures": old_data["T_s"],
            "ground_state_expected_values": old_data["N_0_avgs"],
            "ground_state_second_moment": old_data["N_0_2_avgs"],
            "ground_state_stds": [
                np.sqrt(float(second_moment) - np.square(float(expected_value)))
                for expected_value, second_moment in zip(
                    old_data["N_0_avgs"], old_data["N_0_2_avgs"],
                )
            ],
            "total_energy_expected_values": old_data["U_tot_avgs"],
        }
    with out_path.open('wt') as fd:
        json.dump(number_of_particles_to_data, fd, indent=4)


if __name__ == "__main__":
    main()
