from typing import Any

import numpy as np

from suite2p.extraction import preprocess


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
