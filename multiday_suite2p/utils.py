import glob
import json
import os

from typing import Any, Optional

import numpy as np
import pirt
import scipy.ndimage
import yaml

from ScanImageTiffReader import ScanImageTiffReader
from skimage.measure import find_contours, regionprops
from suite2p.run_s2p import default_ops


def create_mask_img(
    masks: list[dict[str, np.ndarray]],
    im_size: list[int],
    field: Optional[str] = None,
    mark_overlap: bool = False,
    contours: bool = False,
    contour_upsampling: int = 1
) -> np.ndarray:
    """Create label images from cell masks with optional overlap marking and contour generation.

    Args:
        masks (list[dict[str, np.ndarray]]): list of mask dictionaries containing 'xpix' and 'ypix' keys.
        im_size (list[int]): [height, width] of output image.
        field (Optional[str], optional): Mask field to use for pixel values (None for mask IDs).
        mark_overlap (bool, optional): Highlight overlapping regions with value 100.
        contours (bool, optional): Generate mask contours instead of filled regions.
        contour_upsampling (int, optional): Scaling factor for contour resolution.

    Returns:
        Label image with mask representations

    Raises:
        ValueError: If both mark_overlap and contours are enabled
    """
    if mark_overlap and contours:
        raise ValueError("Cannot combine mark_overlap with contour generation")

    # Initialize output image with appropriate dtype
    if (not field) or (field == "id"):
        im = np.zeros(
            (im_size[0] * contour_upsampling, im_size[1] * contour_upsampling),
            dtype=np.uint32,
        )
    else:
        im = np.zeros(im_size, dtype=np.float64)

    for mask_id, mask in enumerate(masks):
        value = mask[field] if field else mask_id
        ypix, xpix = mask["ypix"], mask["xpix"]

        if not contours:
            im[ypix, xpix] = value
            if mark_overlap:
                im[ypix[mask['overlap']], xpix[mask['overlap']]] = 100
        else:
            origin = [min(ypix-1), min(xpix-1)]
            y_local = ypix - origin[0]
            x_local = xpix - origin[1]

            temp_img = np.zeros((max(y_local)+2, max(x_local)+2), dtype=bool)
            temp_img[y_local, x_local] = True
            temp_img = scipy.ndimage.zoom(temp_img, contour_upsampling, order=0)

            contours_ind = np.vstack(find_contours(temp_img)).astype(int)
            im[
                contours_ind[:,0] + (origin[0]*contour_upsampling),
                contours_ind[:,1] + (origin[1]*contour_upsampling)
            ] = value

    return im

def tif_metadata(image_path: str) -> dict[str, Any]:
    """Extract and parse metadata from ScanImage TIFF files.

    Args:
        image_path (str): Path to TIFF file.

    Returns:
        dict[str, Any]: dictionary containing parsed metadata with nested structure.
    """
    image = ScanImageTiffReader(image_path)
    metadata_raw = image.metadata()

    # Split metadata sections
    metadata_str, metadata_json = metadata_raw.split('\n\n', 1)

    # Parse key-value pairs
    metadata_dict = {}
    for item in metadata_str.split('\n'):
        if 'SI.' in item:
            key, val = item.split('=', 1)
            clean_key = key.strip().replace('SI.', '')
            metadata_dict[clean_key] = val.strip()

    # Create nested structure for dotted keys
    for key in list(metadata_dict.keys()):
        if '.' in key:
            parts = key.split('.')
            current = metadata_dict
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = metadata_dict.pop(key)

    # Add JSON metadata and image shape
    metadata_dict['json'] = json.loads(metadata_json)
    metadata_dict['image_shape'] = image.shape()

    return metadata_dict


def metadata_to_ops(metadata: dict[str, Any]) -> dict[str, Any]:
    """Convert ScanImage metadata to Suite2p operations dictionary.

    Args:
        metadata (dict[str, Any]): dictionary from tif_metadata().

    Returns:
        dict[str, Any]: dictionary of Suite2p parameters derived from metadata.
    """
    ops_data: dict[str, Any] = {
        'fs': float(metadata['hRoiManager']['scanVolumeRate']),
        'nplanes': 1 if isinstance(metadata['hFastZ']['userZs'], str) 
                  else len(metadata['hFastZ']['userZs']),
        'nrois': len(metadata['json']['RoiGroups']['imagingRoiGroup']['rois']),
        'nchannels': int(metadata['hChannels']['channelsActive'])
    }

    roi_metadata = metadata['json']['RoiGroups']['imagingRoiGroup']['rois']
    roi_info = {
        'w_px': [],
        'h_px': [],
        'cXY': [],
        'szXY': []
    }

    for roi in roi_metadata:
        scanfields = roi['scanfields']
        roi_info['w_px'].append(scanfields['pixelResolutionXY'][0])
        roi_info['h_px'].append(scanfields['pixelResolutionXY'][1])
        roi_info['cXY'].append(scanfields['centerXY'])
        roi_info['szXY'].append(scanfields['sizeXY'])

    # Calculate spatial parameters
    szXY = np.array(roi_info['szXY'])
    cXY = np.array(roi_info['cXY']) - szXY/2
    cXY -= np.amin(cXY, axis=0)
    mu = np.median(np.column_stack((roi_info['w_px'], roi_info['h_px'])) / szXY, axis=0)
    imin = np.ceil(cXY * mu)

    # Line calculation
    h_px = np.array(roi_info['h_px'])
    n_rows_sum = h_px.sum()
    n_flyback = (metadata['image_shape'][1] - n_rows_sum) / max(1, ops_data['nrois']-1)
    irow = np.insert(np.cumsum(h_px + n_flyback), 0, 0)[:-1]

    # Final ops parameters
    ops_data.update({
        'dx': imin[:,0].astype(np.int32),
        'dy': imin[:,1].astype(np.int32),
        'lines': [np.arange(start, start+h, dtype=np.int32) 
                for start, h in zip(irow, h_px)]
    })

    return ops_data

def yaml_to_dict(file_path: str) -> dict[str, Any]:
    """Load YAML configuration file into dictionary.
    
    Args:
        file_path (str): Path to YAML file.

    Returns:
        dict[str, Any]: Parsed YAML content as dictionary.
    """
    with open(file_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def multiday_ops(
    exp: dict[str, Any],
    session: dict[str, Any],
    folder_name: str,
    settings: dict[str, Any]
) -> dict[str, Any]:
    """Generate Suite2p ops dictionary for multi-day experiments.
    
    Args:
        exp (dict[str, Any]): Experiment configuration dictionary.
        session (dict[str, Any]): Session metadata dictionary.
        folder_name (str): Output directory name.
        settings (dict[str, Any]): Custom Suite2p settings.

    Returns:
        dict[str, Any]: Complete Suite2p operations dictionary.
    """
    data_path = os.path.join(exp['data']['folder_linux'], session['date'], str(session['sub_dir']))
    tif_files = glob.glob(os.path.join(data_path, f"*{session['date']}_{session['sub_dir']}_*.tif"))

    if not tif_files:
        raise FileNotFoundError(f"No TIFF files found in {data_path}")

    ops = metadata_to_ops(tif_metadata(tif_files[0]))
    ops.update({
        'data_path': [data_path],
        'save_path0': data_path,
        'look_one_level_down': False,
        'save_folder': folder_name,
        'fast_disk': os.path.join(data_path, folder_name)
    })

    return {**default_ops(), **settings, **ops}

def create_cropped_deform_field(
    deform: pirt.DeformationFieldBackward,
    origin: np.ndarray,
    crop_size: list[int]
) -> tuple[pirt.DeformationFieldBackward, np.ndarray]:
    """Create cropped deformation field from larger deformation field.
    
    Args:
        deform (pirt.DeformationFieldBackward): Original deformation field.
        origin (np.ndarray): [y, x] coordinates of crop origin.
        crop_size (list[int]): [height, width] of crop region.

    Returns:
        tuple[pirt.DeformationFieldBackward, np.ndarray]: 
            Cropped deformation field and adjusted origin coordinates.
    """
    origin = np.clip(origin, 0, None)
    crop_size = np.array(crop_size)
    im_size = deform[0].shape

    # Adjust origin if crop exceeds image bounds
    for dim in (0, 1):
        if origin[dim] + crop_size[dim] > im_size[dim]:
            origin[dim] = im_size[dim] - crop_size[dim]

    # Create cropped field
    y_slice = slice(origin[0], origin[0]+crop_size[0])
    x_slice = slice(origin[1], origin[1]+crop_size[1])
    return pirt.DeformationFieldBackward([
        deform[0][y_slice, x_slice],
        deform[1][y_slice, x_slice]
    ]), origin

def deform_masks(
    masks: list[dict[str, np.ndarray]],
    deform: pirt.DeformationFieldBackward,
    crop_bin: int = 500
) -> list[dict[str, np.ndarray]]:
    """Apply deformation field to cell masks with local processing.
    
    Args:
        masks (list[dict[str, np.ndarray]]): list of mask dictionaries.
        deform (pirt.DeformationFieldBackward): Deformation field to apply.
        crop_bin (int, optional): Processing window size for memory efficiency.

    Returns:
        list[dict[str, np.ndarray]]: list of deformed masks with updated coordinates.
    """
    deformed = []
    for mask in masks:
        # Local processing window
        crop_origin = np.array(mask["med"], int) - crop_bin//2
        crop_def, adj_origin = create_cropped_deform_field(deform, crop_origin, [crop_bin]*2)

        # Process lambda weights
        y_local = mask["ypix"] - adj_origin[0]
        x_local = mask["xpix"] - adj_origin[1]
        lam_img = np.zeros((crop_bin, crop_bin), dtype=float)
        lam_img[y_local, x_local] = mask["lam"]

        # Apply deformation
        warped_lam = np.array(crop_def.apply_deformation(
            pirt.Aarray(lam_img, origin=tuple(adj_origin)),
            interpolation=0
        ))

        # Extract deformed coordinates
        y_new, x_new = np.nonzero(warped_lam)
        lam_values = warped_lam[y_new, x_new]
        y_global = y_new + adj_origin[0]
        x_global = x_new + adj_origin[1]

        deformed.append({
            'xpix': x_global,
            'ypix': y_global,
            'ipix': np.ravel_multi_index((y_global, x_global), deform[0].shape),
            'med': [np.median(y_global), np.median(x_global)],
            'lam': lam_values,
            'radius': regionprops(warped_lam.astype(np.uint8))[0].minor_axis_length
        })

    return add_overlap_info(deformed)

def add_overlap_info(masks: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
    """Identify overlapping pixels across masks.
    
    Args:
        masks (list[dict[str, np.ndarray]]): list of mask dictionaries with 'ipix' keys.

    Returns:
        list[dict[str, np.ndarray]]: Masks with added 'overlap' boolean arrays.
    """
    all_ipix = np.concatenate([m["ipix"] for m in masks])
    unique, counts = np.unique(all_ipix, return_counts=True)

    for mask in masks:
        mask['overlap'] = np.isin(mask['ipix'], unique[counts > 1])

    return masks
