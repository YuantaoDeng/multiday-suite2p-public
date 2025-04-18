"""This module provides functions for registering, transforming, clustering,
and averaging cell masks across multiple imaging sessions. It uses diffeomorphic
demons registration and clustering based on Jaccard distance to identify
putative cells across sessions, and supports forward and backward transformations of masks.
"""

from typing import Any

import numpy as np
import pirt
import scipy.cluster.hierarchy
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from .utils import add_overlap_info, create_mask_img, deform_masks


def transform_points(
    xpix: np.ndarray,
    ypix: np.ndarray,
    deform: Any
) -> np.ndarray:
    """
    Transform points (in pixel space) according to a DeformationField object.

    Args:
        xpix (np.ndarray): 1D array of x pixel locations.
        ypix (np.ndarray): 1D array of y pixel locations.
        deform (DeformationField): Deformation field object.

    Returns:
        np.ndarray: Transformed point list (num_points x 2) in (y, x) order.
    """
    pp = np.vstack([xpix, ypix]).T.astype(np.float64)
    v = deform.get_field_in_points(pp, 1)
    pp[:, 0] -= v
    v = deform.get_field_in_points(pp, 0)
    pp[:, 1] -= v
    return pp[:, [1, 0]]  # y, x order


def register_sessions(
    images: list[dict[str, np.ndarray]],
    settings: dict[str, Any]
) -> tuple[list[Any], list[dict[str, np.ndarray]]]:
    """
    Registers session images using DiffeomorphicDemonsRegistration and returns deformation objects.

    Args:
        images (list[dict[str, np.ndarray]]): list of session image dictionaries.
        settings (dict[str, Any]): Registration settings.

    Returns:
        tuple[list[Any], list[dict[str, np.ndarray]]]:
            - list of DeformationField objects (one per session).
            - list of transformed images per session.
    """
    ims = [im[settings['img_type']] for im in images]
    reg = pirt.DiffeomorphicDemonsRegistration(*ims)
    reg.params.grid_sampling_factor = settings['grid_sampling_factor']
    reg.params.scale_sampling = settings['scale_sampling']
    reg.params.speed_factor = settings['speed_factor']
    reg.register(verbose=0)
    deforms: list[Any] = []
    trans_images: list[dict[str, np.ndarray]] = []
    for isession in range(len(images)):
        deform = reg.get_deform(isession)
        deforms.append(deform)
        transformed: dict[str, np.ndarray] = {}
        for field in ["mean_img", "enhanced_img", "max_img"]:
            transformed[field] = deform.apply_deformation(images[isession][field])
        trans_images.append(transformed)
    return deforms, trans_images

def transform_cell_masks(
    deforms: list[Any],
    masks: list[list[dict[str, Any]]]
) -> tuple[list[list[dict[str, Any]]], list[np.ndarray]]:
    """
    Transforms cell masks using deformation fields from register_sessions.

    Args:
        deforms (list[Any]): list of DeformationField objects (one per session).
        masks (list[list[dict[str, Any]]]): list of detected cell mask dictionaries per session.

    Returns:
        tuple[list[list[dict[str, Any]]], list[np.ndarray]]:
            - Transformed cell mask dictionaries per session.
            - Transformed labeled mask images per session.
    """
    im_size = deforms[0].field_shape
    trans_masks: list[list[dict[str, Any]]] = []
    trans_label: list[np.ndarray] = []
    for isession, deform in tqdm(enumerate(deforms), total=len(deforms)):
        session_masks = deform_masks(masks[isession], deform)
        # Add session number and set id to 0 (unassigned/not clustered)
        session_masks = [dict(item, **{'session': isession, 'id': 0}) for item in session_masks]
        trans_masks.append(session_masks)
        trans_label.append(create_mask_img(session_masks, im_size, mark_overlap=True))
    return trans_masks, trans_label

def square_to_condensed(i: int, j: int, n: int) -> int:
    """
    Converts squareform indices to condensed form index.

    Args:
        i (int): Index 1.
        j (int): Index 2.
        n (int): Number of entries (length of one dimension of the squareform matrix).

    Returns:
        int: Index into condensed distance matrix.

    Raises:
        AssertionError: If i == j (no diagonal elements in condensed matrix).
    """
    assert i != j, "no diagonal elements in condensed matrix"
    if i < j:
        i, j = j, i
    return int(n * j - j * (j + 1) / 2 + i - 1 - j)

def cluster_cell_masks(
    masks: list[list[dict[str, Any]]],
    im_size: tuple[int, int],
    settings: dict[str, Any],
    verbose: bool = True
) -> tuple[list[list[dict[str, Any]]], np.ndarray]:
    """
    Clusters cell masks across sessions using Jaccard distance matrix.

    Args:
        masks (list[list[dict[str, Any]]]): All cell mask information per session.
        im_size (tuple[int, int]): Image size (height, width).
        settings (dict[str, Any]):
            Clustering settings. Keys include,
            - min_distance: Minimum distance between cell masks to be considered for clustering.
            - criterion: Criterion used for clustering (default: "distance").
            - threshold: Threshold used for clustering (default: {0.975}).
            - min_sessions: Exclude masks not present for this number of times (default: {2}).
            - step_sizes: Clustering happens in these sizes blocks across the plane (for memory reasons).
            - bin_size: Look for masks around center+bin-size to avoid edge cases.
            - min_distance: Only masks with centers within this pixel radius of each other are considered for clustering.
        verbose (bool, optional): If True, show progress. Defaults to True.

    Returns:
        tuple[list[list[dict[str, Any]]], np.ndarray]:
            - list of putative cell masks (each is a list of clustered cell masks).
            - Image of cell masks (label image).
    """
    putative_cells: list[list[dict[str, Any]]] = []
    counter = 0
    for ypos in tqdm(range(0, im_size[0], settings['step_sizes'][1]), disable=not verbose):
        for xpos in range(0, im_size[1], settings['step_sizes'][0]):
            # Collect unassigned masks in range
            cell_info = np.array([
                cell for session in masks for cell in session
                if (cell["id"] == 0)
                and (cell["med"][0] > ypos - settings['bin_size'])
                and (cell["med"][1] > xpos - settings['bin_size'])
                and (cell["med"][0] < ypos + settings['step_sizes'][0] + settings['bin_size'])
                and (cell["med"][1] < xpos + settings['step_sizes'][1] + settings['bin_size'])
            ])
            num_cells = len(cell_info)
            if num_cells > 0:
                centers = np.array([cell["med"] for cell in cell_info])
                dist = np.triu(squareform(pdist(centers) < settings['min_distance']))
                is_possible_pair = np.array(np.where(dist)).T
                if is_possible_pair.shape[0] > 0:
                    jac_shape = int(((num_cells * num_cells) / 2) - (num_cells / 2))
                    jac_mat = np.ones(jac_shape) * 10000
                    for pair in is_possible_pair:
                        if cell_info[pair[0]]["session"] != cell_info[pair[1]]["session"]:
                            num_both = np.intersect1d(
                                cell_info[pair[0]]["ipix"],
                                cell_info[pair[1]]["ipix"],
                                assume_unique=True
                            ).shape[0]
                            jac_mat[square_to_condensed(pair[0], pair[1], num_cells)] = (
                                1 - (num_both / (
                                    cell_info[pair[0]]["ipix"].shape[0]
                                    + cell_info[pair[1]]["ipix"].shape[0]
                                    - num_both
                                ))
                            )
                    Z = scipy.cluster.hierarchy.complete(jac_mat)
                    clust = scipy.cluster.hierarchy.fcluster(
                        Z,
                        t=settings['threshold'],
                        criterion=settings['criterion']
                    )
                    uni, counts = np.unique(clust, return_counts=True)
                    min_sessions = int(np.ceil((settings['min_sessions_perc'] / 100) * len(masks)))
                    clust[np.isin(clust, uni[counts < min_sessions])] = 0
                    uni = np.unique(clust)
                    for clust_id in uni:
                        if clust_id != 0:
                            idx = clust == clust_id
                            med = centers[idx].mean(axis=0)
                            if (
                                (med[0] >= ypos) and (med[0] < ypos + settings['step_sizes'][0])
                                and (med[1] >= xpos) and (med[1] < xpos + settings['step_sizes'][1])
                            ):
                                counter += 1
                                adj_cell_info = []
                                for cell in cell_info[idx]:
                                    new_cell = dict(cell)
                                    new_cell["id"] = counter
                                    adj_cell_info.append(new_cell)
                                putative_cells.append(adj_cell_info)
    # Create result label images
    label_im = np.zeros([len(masks), im_size[0], im_size[1]], np.uint32)
    for masks_group in putative_cells:
        for mask in masks_group:
            label_im[[mask["session"] for _ in mask["xpix"]], mask["ypix"], mask["xpix"]] = masks_group[0]["id"]
    return putative_cells, label_im

def create_template_masks(
    putative_cells: list[list[dict[str, Any]]],
    im_size: tuple[int, int],
    settings: dict[str, Any]
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """
    Create averaged template mask for each group of clustered cell masks (putative cells).

    Args:
        putative_cells (list[list[dict[str, Any]]]): list of putative cell masks.
        im_size (tuple[int, int]): Image plane size (height, width).
        settings (dict[str, Any]): Template mask settings.

    Returns:
        tuple[list[dict[str, Any]], np.ndarray]:
            - list of template mask information.
            - Image of template cell masks.
    """
    template_masks: list[dict[str, Any]] = []
    for masks in putative_cells:
        idx = np.hstack([mask["ipix"] for mask in masks])
        lam = np.hstack([mask["lam"] for mask in masks])
        unique, counts = np.unique(idx, return_counts=True)
        filt_idx = unique[(counts / len(masks)) > (settings['min_perc'] / 100)]
        pixs = np.unravel_index(filt_idx, im_size)
        xpix = pixs[1]
        ypix = pixs[0]
        med = [np.median(ypix), np.median(xpix)]
        radius = np.asarray([mask['radius'] for mask in masks]).mean()
        avg_lem = [lam[idx == i].mean() for i in filt_idx]
        template_masks.append({
            "id": masks[0]["id"],
            "ipix": filt_idx,
            "xpix": xpix,
            "ypix": ypix,
            "med": med,
            "lam": np.array(avg_lem),
            "radius": radius,
            "num_sessions": len(masks)
        })
    template_masks = add_overlap_info(template_masks)
    before_size = len(template_masks)
    template_masks = [
        mask for mask in template_masks
        if (len(mask['ipix']) - sum(mask['overlap'])) >= settings['min_size_non_overlap']
    ]
    print(f"Before filtering: #{before_size} cells, after: #{len(template_masks)} cells")
    template_im = create_mask_img(template_masks, im_size, mark_overlap=True)
    return template_masks, template_im

def backward_transform_masks(
    templates: list[dict[str, Any]],
    deforms: list[Any]
) -> tuple[list[list[dict[str, Any]]], list[np.ndarray], list[np.ndarray]]:
    """
    Perform backward transform of cell masks back to original sample space (unregistered).

    Args:
        templates (list[dict[str, Any]]): list of filtered template masks.
        deforms (list[Any]): list of registration DeformationField objects (one per session).

    Returns:
        tuple[list[list[dict[str, Any]]], list[np.ndarray], list[np.ndarray]]:
            - Cell mask information per session.
            - list of images of cell ids per session.
            - list of images of lambda weights per session.
    """
    trans_masks: list[list[dict[str, Any]]] = []
    deform_lam_imgs: list[np.ndarray] = []
    deform_label_imgs: list[np.ndarray] = []
    im_size = deforms[0][0].shape
    for deform in tqdm(deforms):
        session_masks = deform_masks(templates, deform.as_backward_inverse())
        trans_masks.append(session_masks)
        deform_label_imgs.append(create_mask_img(session_masks, im_size, mark_overlap=True))
        deform_lam_imgs.append(create_mask_img(session_masks, im_size, field="lam"))
    return trans_masks, deform_label_imgs, deform_lam_imgs
