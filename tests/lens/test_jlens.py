"""Pytest for the Jaccard distance LENS function."""

# Author: Matteo Becchi <bechmath@gmail.com>
# Original code by Martina Crippa
# Date: November, 7, 2024

import os
from pathlib import Path
from typing import Generator

import h5py
import numpy as np
import pytest

import dynsight


@pytest.fixture
def original_wd() -> Generator[Path, None, None]:
    original_dir = Path.cwd()

    # Ensure the original working directory is restored after the test
    yield original_dir

    os.chdir(original_dir)


def test_jlens_signals(original_wd: Path) -> None:
    """Test the consistency of jLENS calculations with a control calculation.

    This test verifies that the jLENS calculation yields the same
    values as a control calculation at different r_cut.

    Control file path:
        - tests/systems/2_particles.hdf5

    Dynsight function tested:
        - dynsight.lens.jaccard_change_in_time()

    r_cuts checked:
        - [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    """
    # Define input and output files
    original_dir = Path(__file__).absolute().parent
    input_file = original_dir / "../systems/2_particles.hdf5"
    output_file = original_dir / "../2_particles_test.hdf5"

    # Define trajectory parameters
    traj_name = "2_particles"
    trajectory = slice(0, 20)

    # Define r_cuts
    lens_cutoffs = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]

    # Create universe for lens calculation
    with h5py.File(input_file, "r") as file:
        tgroup = file["Trajectories"][traj_name]
        universe = dynsight.hdf5er.create_universe_from_slice(
            tgroup, trajectory
        )

    # Run jLENS calculation for different r_cuts
    for i in range(len(lens_cutoffs)):
        neig_counts = dynsight.lens.list_neighbours_along_trajectory(
            universe, cutoff=lens_cutoffs[i]
        )
        test_jlens, *_ = dynsight.lens.jaccard_change_in_time(neig_counts)

        # Load check array
        check_jlens = np.load(
            original_wd / f"tests/lens/jLENS_output/jLENS_check_array_{i}.npy"
        )

        # Save output file
        with h5py.File(output_file, "w") as out_file:
            out_file.create_group(f"jLENS_test_{i}")
            out_file[f"jLENS_test_{i}"].create_dataset(
                f"jLENS_test_{i}", data=test_jlens
            )

        # Check if control and test array are equal
        assert np.array_equal(check_jlens, test_jlens), (
            f"jLENS analyses provided different values "
            f"compared to the control system "
            f"for r_cut: {lens_cutoffs[i]} (results: {output_file})."
        )
        # If test passed remove test_jlens array from test folder
        output_file.unlink()
