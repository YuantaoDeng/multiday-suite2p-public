"""This module provides functions for registering, transforming, clustering,
and averaging cell masks across multiple imaging sessions. It uses diffeomorphic
demons registration and clustering based on Jaccard distance to identify
putative cells across sessions, and supports forward and backward transformations of masks.
"""

from typing import Any

import numpy as np
import SimpleITK as sitk
import scipy.cluster.hierarchy
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from .utils import add_overlap_info, create_mask_img, deform_masks


def smooth_and_resample(image, shrink_factors, smoothing_sigmas):
  """
  Smooth and downsample an image by given factors.
  Args:
    image (sitk.Image): The image to resample.
    shrink_factors (int or list): Shrink factor(s) > 1. If scalar, applies to all dimensions.
    smoothing_sigmas (float or list): Gaussian smoothing sigma(s) in physical units. If scalar, applies to all dims.
  Returns:
    sitk.Image: The smoothed and resampled image.
  """
  dim = image.GetDimension()
  # Ensure factors and sigmas are list-like
  if np.isscalar(shrink_factors):
    shrink_factors = [shrink_factors] * dim
  if np.isscalar(smoothing_sigmas):
    smoothing_sigmas = [smoothing_sigmas] * dim
  # Smooth the image
  smoothed = sitk.SmoothingRecursiveGaussian(image, smoothing_sigmas)
  # Calculate new image size and spacing
  orig_size = np.array(image.GetSize(), float)
  orig_spacing = np.array(image.GetSpacing(), float)
  new_size = [int(np.floor(sz / sf + 0.5)) for sz, sf in zip(orig_size, shrink_factors)]
  new_spacing = [((orig_sz - 1) * orig_sp) / (new_sz - 1) if new_sz > 1 else orig_sp 
                 for orig_sz, orig_sp, new_sz in zip(orig_size, orig_spacing, new_size)]
  # Resample the image to the new size
  return sitk.Resample(smoothed, new_size, sitk.Transform(), sitk.sitkLinear,
                       image.GetOrigin(), new_spacing, image.GetDirection(), 0.0, image.GetPixelID())

def multiscale_demons_2d(registration_algorithm, fixed_image, moving_image,
                         initial_transform=None, shrink_factors=None, smoothing_sigmas=None):
  """
  Run a multi-scale demons registration using SimpleITK.
  Args:
    registration_algorithm: Configured SimpleITK demons registration filter (with Execute method).
    fixed_image (sitk.Image): Fixed image (reference) to which moving_image will be registered.
    moving_image (sitk.Image): Moving image that will be transformed.
    initial_transform (sitk.Transform or None): Initial transform for displacement field initialization.
    shrink_factors (list or None): Pyramid shrink factors for each level (excluding full resolution).
    smoothing_sigmas (list or None): Smoothing sigmas for each level (excluding full resolution), in physical units.
  Returns:
    sitk.DisplacementFieldTransform: Resulting displacement field transform mapping fixed -> moving.
  """
  # Pyramid generator
  def image_pair_generator(fixed, moving, factors, sigmas):
    if factors is None or sigmas is None:
      # No multi-resolution, just yield original images
      yield (fixed, moving)
    else:
      for sf, ss in zip(factors, sigmas):
        yield (smooth_and_resample(fixed, sf, ss), smooth_and_resample(moving, sf, ss))
      # Finally yield full resolution images
      yield (fixed, moving)

  # Initialize displacement field at lowest resolution
  if shrink_factors:
    # Determine initial field size/spacing for first level
    sf0 = shrink_factors[0] if isinstance(shrink_factors[0], (list, tuple)) else [shrink_factors[0]] * fixed_image.GetDimension()
    orig_size = np.array(fixed_image.GetSize(), float)
    orig_spacing = np.array(fixed_image.GetSpacing(), float)
    df_size = [int(np.floor(sz / sf + 0.5)) for sz, sf in zip(orig_size, sf0)]
    df_spacing = [((orig_sz - 1) * orig_sp) / (df_sz - 1) if df_sz > 1 else orig_sp 
                  for orig_sz, orig_sp, df_sz in zip(orig_size, orig_spacing, df_size)]
  else:
    df_size = fixed_image.GetSize()
    df_spacing = fixed_image.GetSpacing()
  # Create initial displacement field image (identity/no displacement)
  if initial_transform:
    disp_field = sitk.TransformToDisplacementField(initial_transform, sitk.sitkVectorFloat64,
                                                  df_size, fixed_image.GetOrigin(), df_spacing, fixed_image.GetDirection())
  else:
    disp_field = sitk.Image(df_size, sitk.sitkVectorFloat64, fixed_image.GetDimension())
    disp_field.SetOrigin(fixed_image.GetOrigin()); disp_field.SetSpacing(df_spacing); disp_field.SetDirection(fixed_image.GetDirection())
  # Multi-resolution registration
  for f_img, m_img in image_pair_generator(fixed_image, moving_image, shrink_factors, smoothing_sigmas):
    disp_field = sitk.Resample(disp_field, f_img)  # upsample field to current level
    disp_field = registration_algorithm.Execute(f_img, m_img, disp_field)
  return sitk.DisplacementFieldTransform(disp_field)


class DemonsTransform:
  """Wrapper for a displacement field transform and associated image shape."""
  def __init__(self, sitk_transform, field_shape):
    self.transform = sitk_transform           # SimpleITK transform (displacement field)
    self.field_shape = field_shape           # (rows, cols) shape of the reference image
  def apply_deformation(self, image_array, interpolator=sitk.sitkLinear):
    """Apply this transform to warp a 2D numpy image to the reference frame."""
    # Convert numpy array to SimpleITK
    sitk_image = sitk.GetImageFromArray(image_array)
    sitk_image.SetOrigin((0.0, 0.0)); sitk_image.SetSpacing((1.0, 1.0))
    # Define reference output image (same size as reference frame)
    out_size = (int(self.field_shape[1]), int(self.field_shape[0]))  # (width, height)
    reference = sitk.Image(out_size, sitk_image.GetPixelID())
    reference.SetOrigin((0.0, 0.0)); reference.SetSpacing((1.0, 1.0)); reference.SetDirection(sitk_image.GetDirection())
    # Resample moving image onto reference grid
    warped = sitk.Resample(sitk_image, reference, self.transform, interpolator, 0.0)
    return sitk.GetArrayFromImage(warped)
  
  
def transform_points(xpix, ypix, deform):
  """Transform an array of points (pixel coordinates) using a given deformation (approximate inverse mapping).
  Args:
    xpix (np.ndarray): X pixel coordinates (1D).
    ypix (np.ndarray): Y pixel coordinates (1D).
    deform (DemonsTransform or sitk.Transform): Deformation mapping reference->moving (backward displacement).
  Returns:
    np.ndarray: Transformed points (N x 2 array) in (y, x) format.
  """
  pts = np.vstack([xpix, ypix]).T.astype(float)
  sitk_tx = deform.transform if hasattr(deform, 'transform') else deform  # underlying SimpleITK transform
  transformed_pts = []
  for (x, y) in pts:
    # We have a backward transform mapping fixed->moving; to get the approximate inverse (moving->fixed),
    # use one iteration of inverse mapping: start at source (moving) point and adjust by error.
    target = np.array([x, y], float)  # initialize guess for fixed coordinate
    mapped = np.array(sitk_tx.TransformPoint((target[0], target[1])))
    error = mapped - np.array([x, y], float)
    target -= error  # adjust guess
    transformed_pts.append([target[1], target[0]])  # (y, x)
  return np.array(transformed_pts)


def register_sessions(images, settings):
  """Registers session images using SimpleITK Demons (multi-scale) and returns transforms and warped images.
  Args:
    images (list of dict): List of session image dicts (each with keys "mean_img", "enhanced_img", "max_img").
    settings (dict): Registration settings with keys:
       'img_type': str, image key to use for registration (e.g. "enhanced_img").
       'grid_sampling_factor': int (not used in SimpleITK approach; kept for compatibility).
       'scale_sampling': int, number of iterations for demons at each scale.
       'speed_factor': int (not directly used; controls Demons aggressiveness in PIRT, ignored here).
  Returns:
    list: List of DemonsTransform objects (length = num_sessions).
    list: List of transformed images per session (each is a dict like input, warped to reference frame).
  """
  img_type = settings.get('img_type', 'enhanced_img')
  # Prepare fixed reference image (session 0)
  fixed_array = images[0][img_type].astype(np.float32)
  fixed_image = sitk.GetImageFromArray(fixed_array)
  fixed_image.SetOrigin((0.0, 0.0)); fixed_image.SetSpacing((1.0, 1.0))
  # Configure demons registration filter
  demon_cfg = settings['demons']
  demons = sitk.DiffeomorphicDemonsRegistrationFilter()
  demons.SetNumberOfIterations(int(demon_cfg['iterations']))
  demons.SetSmoothDisplacementField(bool(demon_cfg['smooth_displacement']))
  demons.SetStandardDeviations(float(demon_cfg['smoothing_sigma']))
  # Define multi-scale (two levels: 1/4 and 1/2 resolutions)
  shrink_factors   = demon_cfg['shrink_factors']
  smoothing_sigmas = demon_cfg['smoothing_sigmas']
  deforms = []
  trans_images = []
  # Reference session (index 0): identity transform
  identity_tx = sitk.Transform(2, sitk.sitkIdentity)
  deforms.append(DemonsTransform(identity_tx, fixed_array.shape))
  trans_images.append({field: images[0][field].copy() for field in ["mean_img", "enhanced_img", "max_img"]})
  # Register other sessions to reference
  for i in range(1, len(images)):
    moving_array = images[i][img_type].astype(np.float32)
    moving_image = sitk.GetImageFromArray(moving_array)
    moving_image.SetOrigin((0.0, 0.0)); moving_image.SetSpacing((1.0, 1.0))
    # Run multi-scale demons registration
    disp_tx = multiscale_demons_2d(demons, fixed_image, moving_image, initial_transform=None,
                                   shrink_factors=shrink_factors, smoothing_sigmas=smoothing_sigmas)
    demons_tx = DemonsTransform(disp_tx, fixed_array.shape)
    deforms.append(demons_tx)
    # Warp all image types of this session to reference frame
    transformed = {}
    for field in ["mean_img", "enhanced_img", "max_img"]:
      transformed[field] = demons_tx.apply_deformation(images[i][field])
    trans_images.append(transformed)
  return deforms, trans_images


def transform_cell_masks(deforms, masks):
  """Transforms cell masks from each session into the reference frame.
  Args:
    deforms (list): List of deformation transforms (length = num_sessions).
    masks (list): List of session mask lists (each element is a list of mask dicts for that session).
  Returns:
    list: Transformed masks for each session (in reference frame).
    list: Labeled mask images per plane (with overlaps marked).
  """
  im_size = deforms[0].field_shape  # shape of reference image
  trans_masks = []
  trans_label = []
  for isession, deform in tqdm(enumerate(deforms), total=len(deforms)):
    session_masks = deform_masks(masks[isession], deform)
    # annotate session index and initialize cluster id to 0
    session_masks = [dict(mask, **{'session': isession, 'id': 0}) for mask in session_masks]
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


def cluster_cell_masks(masks, im_size, settings, verbose=True):
  """Clusters cell masks across sessions using a Jaccard distance matrix.
  Args:
    masks (list of lists): All cell masks grouped by session (length = num_sessions).
      Each mask is a dict with 'xpix','ypix','ipix','lam','med','session','plane','id'.
    im_size (tuple): [height, width] of image plane.
    settings (dict): Clustering parameters:
      - min_distance (float): minimum Euclidean distance between mask centers to consider clustering.
      - criterion (str): criterion for hierarchical clustering (default "distance").
      - threshold (float): threshold for clustering (default 0.975 for Jaccard distance).
      - min_sessions_perc (int): minimum % of sessions a mask must appear in (for cluster filtering).
      - step_sizes (list): block sizes (Y, X) for iterative clustering to save memory.
      - bin_size (int): margin around block to include nearby masks.
      - min_distance (int): (repeated key in settings) radius for initial center proximity filter.
  Returns:
    list: List of putative cell groups (each group is a list of masks across sessions belonging to one putative cell).
    np.ndarray: Label image marking clustered masks (per putative cell) across sessions.
  """
  putative_cells = []
  counter = 0
  # Iterate through image in blocks to cluster locally
  for ypos in tqdm(range(0, im_size[0], settings['step_sizes'][1]), disable=(not verbose)):
    for xpos in range(0, im_size[1], settings['step_sizes'][0]):
      # collect unassigned masks in this block (id == 0 means not yet clustered)
      cell_info = np.array([cell for session_masks in masks for cell in session_masks
                             if cell["id"] == 0 
                             and (cell["med"][0] > ypos - settings['bin_size']) 
                             and (cell["med"][1] > xpos - settings['bin_size']) 
                             and (cell["med"][0] < ypos + settings['step_sizes'][0] + settings['bin_size']) 
                             and (cell["med"][1] < xpos + settings['step_sizes'][1] + settings['bin_size'])])
      num_cells = len(cell_info)
      if num_cells == 0:
        continue
      # Compute pairwise center distances and identify close pairs
      centers = np.array([cell["med"] for cell in cell_info])
      dist = np.triu(squareform(pdist(centers) < settings['min_distance']))
      is_possible_pair = np.array(np.where(dist)).T
      # Calculate Jaccard distances for each potential pair
      if is_possible_pair.shape[0] > 0:
        jac_shape = int((num_cells * (num_cells - 1)) / 2)
        jac_mat = np.ones(jac_shape) * 10000  # initialize large distances
        for pair in is_possible_pair:
          if cell_info[pair[0]]["session"] == cell_info[pair[1]]["session"]:
            continue  # skip pair from same session
          # Jaccard distance = 1 - (intersection / union) of pixels
          num_both = np.intersect1d(cell_info[pair[0]]["ipix"], cell_info[pair[1]]["ipix"], assume_unique=True).shape[0]
          jac_index = square_to_condensed(pair[0], pair[1], num_cells)
          jac_mat[jac_index] = 1 - (num_both / (cell_info[pair[0]]["ipix"].size + cell_info[pair[1]]["ipix"].size - num_both))
        # Hierarchical clustering (complete linkage)
        Z = scipy.cluster.hierarchy.complete(jac_mat)
        clust = scipy.cluster.hierarchy.fcluster(Z, t=settings['threshold'], criterion=settings['criterion'])
        # Exclude clusters that appear in fewer than min_sessions_perc of sessions
        unique_clusters, counts = np.unique(clust, return_counts=True)
        min_sessions = int(np.ceil((settings['min_sessions_perc'] / 100) * len(masks)))
        for cid, count in zip(unique_clusters, counts):
          if count < min_sessions:
            clust[clust == cid] = 0  # mark as noise (0)
        # Record clusters that fall entirely within this block
        unique_clust = np.unique(clust)
        for clust_id in unique_clust:
          if clust_id == 0:
            continue
          # Compute cluster centroid and ensure it lies in current block (not just fringe)
          idx = (clust == clust_id)
          med = centers[idx].mean(axis=0)
          if (med[0] >= ypos) and (med[0] < ypos + settings['step_sizes'][0]) \
             and (med[1] >= xpos) and (med[1] < xpos + settings['step_sizes'][1]):
            counter += 1
            # Assign a new cluster id and mark masks
            cluster_masks = []
            for cell in cell_info[idx]:
              new_cell = dict(cell)  # copy to avoid modifying original
              new_cell["id"] = counter
              cluster_masks.append(new_cell)
            putative_cells.append(cluster_masks)
  # Create label image stack (dimensions: num_sessions x H x W)
  label_im = np.zeros((len(masks), im_size[0], im_size[1]), np.uint32)
  for cell_group in putative_cells:
    cell_id = cell_group[0]["id"]
    for mask in cell_group:
      # mark all pixels of this mask with its cell_id in the corresponding session layer
      label_im[mask["session"], mask["ypix"], mask["xpix"]] = cell_id
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


def backward_transform_masks(templates, deforms):
  """Transform template masks from reference frame back to each original session's space.
  Args:
    templates (list): List of template mask dicts (consensus masks in reference frame).
    deforms (list): List of DemonsTransform or sitk.Transform for each session (as from register_sessions).
  Returns:
    list: List of length num_sessions, each an embedded list of mask dicts (templates mapped to that session).
    list: List of label images (per session) with cell IDs.
    list: List of images (per session) with lambda weights.
  """
  trans_masks = []
  label_imgs = []
  lam_imgs = []
  # Determine image size (reference image shape)
  if hasattr(deforms[0], 'field_shape'):
    im_shape = deforms[0].field_shape
  else:
    raise ValueError("Transforms do not contain field_shape information.")
  for s_idx, deform in enumerate(tqdm(deforms, desc="Backward transforming masks")):
    sitk_tx = deform.transform if hasattr(deform, 'transform') else deform
    session_masks = []
    for tmpl in templates:
      # Transform each pixel of template mask to this session
      xpix = tmpl['xpix']; ypix = tmpl['ypix']; lam = tmpl['lam']
      coords = [sitk_tx.TransformPoint((float(x), float(y))) for x, y in zip(xpix, ypix)]
      coords = np.array(coords)
      # Round to nearest pixel
      new_x = np.rint(coords[:,0]).astype(int)
      new_y = np.rint(coords[:,1]).astype(int)
      # Filter points outside the image bounds
      valid = (new_x >= 0) & (new_x < im_shape[1]) & (new_y >= 0) & (new_y < im_shape[0])
      if not np.any(valid):
        continue
      new_x = new_x[valid]; new_y = new_y[valid]; lam_vals = lam[valid]
      ipix = np.ravel_multi_index((new_y, new_x), (im_shape[0], im_shape[1]))
      med = [float(np.median(new_y)), float(np.median(new_x))]
      info = {
        'xpix': new_x,
        'ypix': new_y,
        'ipix': ipix,
        'lam': lam_vals,
        'med': med,
        'id': tmpl.get('id', 0)
      }
      session_masks.append(info)
    session_masks = add_overlap_info(session_masks)
    trans_masks.append(session_masks)
    label_imgs.append(create_mask_img(session_masks, im_shape, mark_overlap=True))
    lam_imgs.append(create_mask_img(session_masks, im_shape, field='lam'))
  return trans_masks, label_imgs, lam_imgs
