import glob
import json
import os

from typing import Any, Optional

import numpy as np
import SimpleITK as sitk
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


def deform_masks(masks, deform, crop_bin=500):
  """Deforms cell masks according to a given deformation (SimpleITK transform).
  Args:
    masks (list of dict): List of mask dicts with 'xpix','ypix','lam'.
    deform (DemonsTransform or sitk.Transform): Transform mapping reference frame -> moving frame (backward field).
    crop_bin (int): Size of the cropping window for warping (pixels).
  Returns:
    list of dict: Deformed masks (with 'xpix','ypix','lam','ipix','med','radius').
  """
  deformed_masks = []
  # Determine image full size from transform
  if hasattr(deform, 'field_shape'):
    full_shape = deform.field_shape  # (height, width)
  else:
    raise ValueError("Deform must have field_shape attribute or known image size.")
  sitk_tx = deform.transform if hasattr(deform, 'transform') else deform
  for mask in masks:
    # Define crop region around mask center
    cy, cx = int(mask['med'][0]), int(mask['med'][1])
    ori_y = max(0, cy - crop_bin // 2)
    ori_x = max(0, cx - crop_bin // 2)
    if ori_y + crop_bin > full_shape[0]:
      ori_y = max(0, full_shape[0] - crop_bin)
    if ori_x + crop_bin > full_shape[1]:
      ori_x = max(0, full_shape[1] - crop_bin)
    crop_h = min(crop_bin, full_shape[0])
    crop_w = min(crop_bin, full_shape[1])
    # Create an image patch of lambda values for this mask (in moving space)
    lam_patch = np.zeros((crop_h, crop_w), float)
    local_y = mask['ypix'] - ori_y
    local_x = mask['xpix'] - ori_x
    valid = (local_y >= 0) & (local_y < crop_h) & (local_x >= 0) & (local_x < crop_w)
    lam_patch[local_y[valid], local_x[valid]] = mask['lam'][valid]
    # Convert to SimpleITK image and position it in moving frame
    sitk_patch = sitk.GetImageFromArray(lam_patch)
    sitk_patch.SetOrigin((float(ori_x), float(ori_y)))
    sitk_patch.SetSpacing((1.0, 1.0))
    # Define output patch in reference frame (same size and origin in reference coords)
    out_img = sitk.Image(crop_w, crop_h, sitk.sitkFloat64)
    out_img.SetOrigin((float(ori_x), float(ori_y)))
    out_img.SetSpacing((1.0, 1.0))
    # Resample from moving to reference using the backward transform
    warped = sitk.Resample(sitk_patch, out_img, sitk_tx, sitk.sitkNearestNeighbor, 0.0)
    warped_arr = sitk.GetArrayFromImage(warped)
    pixs = np.argwhere(warped_arr != 0)
    if pixs.size == 0:
      continue  # no pixels (mask may have moved out of frame)
    lam_vals = warped_arr[pixs[:,0], pixs[:,1]]
    pixs_global = pixs + np.array([ori_y, ori_x])
    pixs_global = pixs_global.astype(int)
    ipix = np.ravel_multi_index((pixs_global[:,0], pixs_global[:,1]), (full_shape[0], full_shape[1]))
    med_new = [float(np.median(pixs_global[:,0])), float(np.median(pixs_global[:,1]))]
    # Compute radius via region properties on warped mask
    mask_bin = (warped_arr > 0).astype(np.uint8)
    props = regionprops(mask_bin)
    radius_new = min([prop.minor_axis_length for prop in props]) if props else 0.0
    deformed_masks.append({
      'xpix': pixs_global[:,1],
      'ypix': pixs_global[:,0],
      'ipix': ipix,
      'med': med_new,
      'lam': lam_vals,
      'radius': radius_new
    })
  deformed_masks = add_overlap_info(deformed_masks)
  return deformed_masks

def add_overlap_info(masks):
  """Adds an 'overlap' boolean array to each mask dict indicating overlapping pixels.
  Args:
    masks (list of dict): Masks with 'ipix' field.
  Returns:
    list of dict: Same list with 'overlap' field added to each mask.
  """
  if not masks:
    return masks
  all_ipix = np.concatenate([mask['ipix'] for mask in masks]).astype(int)
  unique_pixels, counts = np.unique(all_ipix, return_counts=True)
  for mask in masks:
    inds = np.searchsorted(unique_pixels, mask['ipix'])
    mask['overlap'] = counts[inds] > 1
  return masks
