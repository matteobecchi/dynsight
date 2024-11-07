from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from MDAnalysis import AtomGroup, Universe

import numpy as np
from MDAnalysis.lib.NeighborSearch import AtomNeighborSearch


def list_neighbours_along_trajectory(
    input_universe: Universe,
    cutoff: float,
    trajslice: slice | None = None,
) -> list[list[AtomGroup]]:
    """Produce a per frame list of the neighbours, atom per atom.

    * Original author: Martina Crippa
    * Mantainer: Daniele Rapetti

    Parameters:
        input_universe (Universe):
            the universe, or the atomgroup containing the trajectory.
        cutoff (float):
            the maximum neighbour distance.
        trajslice (slice, optional):
            the slice of the trajectory to consider. Defaults to slice(None).

    Returns:
        list[list[AtomGroup]]:
            list of AtomGroup wint the neighbours of each atom for each frame
    """
    if trajslice is None:
        trajslice = slice(None)
    neigh_list_per_frame = []
    for _ in input_universe.universe.trajectory[trajslice]:
        neigh_search = AtomNeighborSearch(
            input_universe.atoms, box=input_universe.dimensions
        )

        neigh_list_per_atom = [
            neigh_search.search(atom, cutoff) for atom in input_universe.atoms
        ]
        neigh_list_per_frame.append([at.ix for at in neigh_list_per_atom])
    return neigh_list_per_frame


def neighbour_change_in_time(
    neigh_list_per_frame: list[list[AtomGroup]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """return, listed per each atoms the parameters used in the LENS analysis.

    * Original author: Martina Crippa
    * Mantainer: Daniele Rapetti

    Parameters:
        neigh_list_per_frame:
             a frame by frame list of the neighbours of each atom output
             of :func:`listNeighboursAlongTrajectory`.

    Returns:
        - **lensArray** The calculated LENS parameter
        - **numberOfNeighs** the count of neighbours per frame
        - **lensNumerators** the numerators used for calculating LENS
            parameter
        - **lensDenominators** the denominators used for calculating LENS
            parameter
    """
    nat = np.asarray(neigh_list_per_frame, dtype=object).shape[1]
    nframes = np.asarray(neigh_list_per_frame, dtype=object).shape[0]
    # this is the number of common NN between frames
    lensarray = np.zeros((nat, nframes))
    # this is the number of NN at that frame
    numberofneighs = np.zeros((nat, nframes))
    # this is the numerator of LENS
    lensnumerators = np.zeros((nat, nframes))
    # this is the denominator of lens
    lensdenominators = np.zeros((nat, nframes))
    # each nnlist contains also the atom that generates them,
    # so 0 nn is a 1 element list
    for atom_id in range(nat):
        numberofneighs[atom_id, 0] = (
            neigh_list_per_frame[0][atom_id].shape[0] - 1
        )
        # let's calculate the numerators and the denominators
        for frame in range(1, nframes):
            numberofneighs[atom_id, frame] = (
                neigh_list_per_frame[frame][atom_id].shape[0] - 1
            )
            lensdenominators[atom_id, frame] = (
                neigh_list_per_frame[frame][atom_id].shape[0]
                + neigh_list_per_frame[frame - 1][atom_id].shape[0]
                - 2
            )
            lensnumerators[atom_id, frame] = np.setxor1d(
                neigh_list_per_frame[frame][atom_id],
                neigh_list_per_frame[frame - 1][atom_id],
            ).shape[0]

    den_not_0 = lensdenominators != 0
    # lens
    lensarray[den_not_0] = (
        lensnumerators[den_not_0] / lensdenominators[den_not_0]
    )
    return lensarray, numberofneighs, lensnumerators, lensdenominators


def jaccard_change_in_time(
    neigh_list_per_frame: list[list[AtomGroup]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Returns the time variation of neighbors using the Jaccard distance.

    * Author: Matteo Becchi <bechmath@gmail.com>
    * Original code by: Martina Crippa
    * Mantainer: Andrew Tarzia
    * Date: November, 7, 2024

    Parameters:
        neigh_list_per_frame:
            a frame by frame list of the neighbours of each atom output
            of :func:`listNeighboursAlongTrajectory`.

    Returns:
        - **jlensArray** The calculated jLENS parameter
        - **jlensNumerators** the numerators used for calculating jLENS
        - **jlensDenominators** the denominators used for calculating jLENS

    Notes:
    Each nnlist contains also the atom that generates them,
    so 0 neighbors is a 1 element list.
    """
    n_atoms = np.asarray(neigh_list_per_frame, dtype=object).shape[1]
    n_frames = np.asarray(neigh_list_per_frame, dtype=object).shape[0]

    jlens_num = np.zeros((n_atoms, n_frames - 1))
    jlens_den = np.zeros((n_atoms, n_frames - 1))
    jlens_array = np.zeros((n_atoms, n_frames - 1))

    for atom_id in range(n_atoms):
        for frame in range(1, n_frames):
            jlens_num[atom_id, frame - 1] = np.setxor1d(
                neigh_list_per_frame[frame][atom_id],
                neigh_list_per_frame[frame - 1][atom_id],
            ).shape[0]
            jlens_den[atom_id, frame - 1] = np.union1d(
                neigh_list_per_frame[frame][atom_id],
                neigh_list_per_frame[frame - 1][atom_id],
            ).shape[0]

    den_not_0 = jlens_den != 0

    jlens_array[den_not_0] = jlens_num[den_not_0] / jlens_den[den_not_0]

    return jlens_array, jlens_num, jlens_den
