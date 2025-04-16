import os
from pathlib import Path
from typing import List, Tuple, Any, Dict, Union

import numpy as np
from suite2p.io import compute_dydx, BinaryFileCombined
from suite2p.extraction.masks import create_masks
from suite2p.extraction.extract import extract_traces

def extract_traces_session(
    multiday_folder: Union[str, Path],
    data_folder: Union[str, Path],
    bin_folder: Union[str, Path],
    data_path: Union[str, Path]
) -> None:
    """
    Extracts fluorescence and neuropil traces for a single session using registered masks.
    This function is designed for parallelization across sessions.

    Args:
        multiday_folder (Union[str, Path]): Path to the multi-day output folder.
        data_folder (Union[str, Path]): Path to the folder containing session data.
        bin_folder (Union[str, Path]): Path to the folder containing binary files for all sessions.
        data_path (Union[str, Path]): Relative path to the session (e.g., 'YYYY_MM_DD/session_id').

    Returns:
        None. Saves extracted traces and ops to the session's output folder.

    Raises:
        NameError: If the session's cell masks cannot be found.
    """
    # Convert to Path objects for consistency
    multiday_folder = Path(multiday_folder)
    data_folder = Path(data_folder)
    bin_folder = Path(bin_folder)
    data_path = Path(data_path)

    # Create or clear the save folder for this session
    save_folder = multiday_folder / 'sessions' / data_path
    save_folder.mkdir(parents=True, exist_ok=True)
    if save_folder.is_dir():
        print(f'\nRemoving files in {save_folder}')
        for f in save_folder.glob('*'):
            os.remove(f)

    print('\nCollecting data')
    # Load session info and suite2p ops for all planes
    info: Dict[str, Any] = np.load(multiday_folder / 'info.npy', allow_pickle=True).item()
    plane_folders: List[Path] = list((bin_folder / data_path / info['suite2p_folder']).glob('plane[0-9]'))
    ops1: List[Dict[str, Any]] = [np.load(plane_folder / 'ops.npy', allow_pickle=True).item() for plane_folder in plane_folders]
    reg_loc: List[Path] = [plane_folder / 'data.bin' for plane_folder in plane_folders]
    dy, dx = compute_dydx(ops1)
    Ly = np.array([ops['Ly'] for ops in ops1])
    Lx = np.array([ops['Lx'] for ops in ops1])
    LY = int(np.amax(dy + Ly))
    LX = int(np.amax(dx + Lx))

    # Find index of cell masks for this session
    session_ind: Union[int, None] = None
    for i, j in enumerate(info['data_paths']):
        if j == data_path.as_posix():
            session_ind = i
            break
    if session_ind is None:
        raise NameError(f'Could not find cell masks for {data_path.as_posix()}')

    # Load cell mask stats for this session
    stats_combined: List[Dict[str, Any]] = np.load(multiday_folder / 'backwards_deformed_cell_masks.npy', allow_pickle=True)[session_ind]
    if 'overlap' not in stats_combined[0]:
        for stat in stats_combined:
            stat['overlap'] = True

    # Load combined ops and set overlap allowance
    ops_file = data_folder / data_path / info['suite2p_folder'] / 'combined' / 'ops.npy'
    ops_combined: Dict[str, Any] = np.load(ops_file, allow_pickle=True).item()
    ops_combined['allow_overlap'] = True

    # Create cell and neuropil masks in the global (stitched) view
    print('\nCreating masks')
    cell_masks, neuropil_masks = create_masks(stats_combined, ops_combined['Ly'], ops_combined['Lx'], ops_combined)

    # Extract traces from the registered binary files
    print('\nExtracting traces')
    with BinaryFileCombined(LY, LX, Ly, Lx, dy, dx, reg_loc) as f:
        F, Fneu = extract_traces(f, cell_masks, neuropil_masks, ops_combined["batch_size"])

    # Save results to disk
    print(f"\nSaving results in {save_folder}..")
    np.save(save_folder / 'ops.npy', ops_combined)
    np.save(save_folder / 'F.npy', F)
    np.save(save_folder / 'Fneu.npy', Fneu)
