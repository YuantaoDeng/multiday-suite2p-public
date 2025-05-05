import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from suite2p.extraction import preprocess
from suite2p.extraction.masks import create_masks
from suite2p.extraction.extract import extract_traces
from suite2p.io import compute_dydx, BinaryFileCombined

from .io import test_extract_result_present


def demix_traces(
    F: np.ndarray,
    Fneu: np.ndarray,
    cell_masks: list[dict[str, Any]],
    ops: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Demix activity from overlapping cells using mask covariance & regularized linear regression.

    Args:
        F (np.ndarray): Raw fluorescence activity (shape: num_cells x num_frames).
        Fneu (np.ndarray): Raw neuropil activity (shape: num_cells x num_frames).
        cell_masks (list[dict[str, Any]]):
            list of cell mask dictionaries (length: num_cells). Each dict must contain keys 'xpix',
            'ypix', 'lam', and 'overlap'.
        ops (dict[str, Any]):
            Parameters for demixing. Must contain keys 'baseline', 'win_baseline', 'sig_baseline',
            'fs', 'neucoeff', 'l2_reg', 'Ly', and 'Lx'.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - Fdemixed (np.ndarray): Demixed fluorescence traces (num_cells x num_frames).
            - Fbase (np.ndarray): Baseline-subtracted fluorescence traces (num_cells x num_frames).
            - covU (np.ndarray): Covariance matrix of cell masks (num_cells x num_cells).
            - lammap (np.ndarray): Weight mask for each cell (num_cells x Ly x Lx).
    """
    # Subtract neuropil signal and baseline
    Fcorr = F - ops['neucoeff'] * Fneu
    Fbase = preprocess(Fcorr, ops['baseline'], ops['win_baseline'],
                       ops['sig_baseline'], ops['fs'])
    # Collect mask information
    num_cells = len(cell_masks)
    Ly, Lx = ops['Ly'], ops['Lx']
    lammap = np.zeros((num_cells, Ly, Lx), dtype=np.float32)  # weight mask for each cell
    Umap = np.zeros((num_cells, Ly, Lx), dtype=bool)  # binarized weight masks
    covU = np.zeros((num_cells, num_cells), dtype=np.float32)  # covariance matrix
    for ni, mask in enumerate(cell_masks):
        ypix, xpix, lam = mask['ypix'], mask['xpix'], mask['lam']
        norm = lam.sum()
        Fbase[ni] *= norm
        lammap[ni, ypix, xpix] = lam
        Umap[ni, ypix, xpix] = True
        covU[ni, ni] = (lam ** 2).sum()
    # Create covariance matrix of the masks
    for ni, mask in enumerate(cell_masks):
        if np.sum(mask['overlap']) > 0:
            ioverlap = mask['overlap']
            yp, xp, lam = mask['ypix'][ioverlap], mask['xpix'][ioverlap], mask['lam'][ioverlap]
            njs, ijs = np.nonzero(Umap[:, yp, xp])
            for nj in np.unique(njs):
                if nj != ni:
                    inds = ijs[njs == nj]
                    covU[ni, nj] = (lammap[nj, yp[inds], xp[inds]] * lam[inds]).sum()
    # Solve for demixed traces of the cells
    l2 = np.diag(covU).mean() * ops['l2_reg']
    Fdemixed = np.linalg.solve(covU + l2 * np.eye(num_cells), Fbase)
    return Fdemixed, Fbase, covU, lammap


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
    info: dict[str, Any] = np.load(multiday_folder / 'info.npy', allow_pickle=True).item()
    plane_folders: list[Path] = list((data_folder / data_path / info['suite2p_folder']).glob('plane[0-9]'))
    ops1: list[dict[str, Any]] = [np.load(plane_folder / 'ops.npy', allow_pickle=True).item() for plane_folder in plane_folders]
    reg_loc: list[Path] = [plane_folder / 'data.bin' for plane_folder in plane_folders]
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
    stats_combined: list[dict[str, Any]] = np.load(multiday_folder / 'backwards_deformed_cell_masks.npy', allow_pickle=True)[session_ind]
    if 'overlap' not in stats_combined[0]:
        for stat in stats_combined:
            stat['overlap'] = True

    # Load combined ops and set overlap allowance
    ops_file = data_folder / data_path / info['suite2p_folder'] / 'combined' / 'ops.npy'
    ops_combined: dict[str, Any] = np.load(ops_file, allow_pickle=True).item()
    ops_combined['allow_overlap'] = True

    # Create cell and neuropil masks in the global (stitched) view
    print('\nCreating masks')
    stats_combined = [stat for stat in stats_combined if stat['radius'] > 0]  # TODO: hack to resolve scipy error
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


def extract_local(
    data_info: dict[str, Any],
    data_path: Union[str, Path],
    force_recalc: bool = False,
) -> Optional[dict[str, Any]]:
    """
    Run an extraction job directly in the current (serverless) Python environment,
    if results are not already present or recalculation is forced.

    Args:
        data_info (dict[str, Any]): Data info dictionary with data and output folder paths.
        data_path (Union[str, Path]): Path to the session (e.g., 'YYYY_MM_DD/session_id').
        force_recalc (bool, optional): If True, force recalculation even if results exist. Defaults to False.

    Returns:
        Optional[dict[str, Any]]: dictionary with 'data_path' and 'job_id' (always None in serverless mode) if job is run, else None.

    Raises:
        NameError: If the extraction fails.
    """
    result_folder = Path(data_info['data']['processed_data_folder']) / data_info['data']['output_folder'] / 'sessions' / data_path
    if (not test_extract_result_present(result_folder)) or force_recalc:
        multiday_folder = (Path(data_info['data']['processed_data_folder']) / data_info['data']['output_folder']).as_posix()
        data_folder = data_info['data']['local_processed_root']
        bin_folder = data_info['data']['local_bin_root']
        data_path = Path(data_path).as_posix()

        print(f"multiday_folder: {multiday_folder}")
        print(f"data_folder: {data_folder}")
        print(f"bin_folder: {bin_folder}")
        print(f"data_path: {data_path}")

        # Directly call the Python extraction function
        try:
            extract_traces_session(multiday_folder, data_folder, bin_folder, data_path)
            print("Extraction completed successfully.")
            return {'data_path': data_path, 'job_id': None}
        except Exception as e:
            print(f"Extraction failed: {e}")
            raise NameError("Extraction failed to run successfully.") from e
    else:
        print(f'{data_path} - Already present')
    return None
