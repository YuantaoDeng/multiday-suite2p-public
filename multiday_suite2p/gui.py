from typing import Any, Union

import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


from IPython.display import display
from ipywidgets import HBox, VBox


def show_imgs_with_masks(
    sessions: list[dict[str, Any]],
    images: list[Union[dict[str, np.ndarray], np.ndarray]],
    mask_sets: dict[str, Union[list[np.ndarray], np.ndarray]],
    aspect_ratio: float = 1.5
) -> None:
    """
    Interactive GUI for visualizing session images and overlaying different sets of cell masks.

    Args:
        sessions (list[dict[str, Any]]):
            list of dictionaries with session info ('date', 'session_id'). Length is number of sessions.
        images (list[Union[dict[str, np.ndarray], np.ndarray]]):
            list of dictionaries (for multiple image types) or arrays (for single image type).
            Each dictionary entry should have keys like 'mean_img', 'enhanced_img', 'max_img'.
        mask_sets (dict[str, Union[list[np.ndarray], np.ndarray]]):
            dictionary containing different images corresponding to cell masks (pixel value is cell mask identity).
            Each value is either a list of label images (one per session) or a 3D array (session x H x W).
        aspect_ratio (float, optional):
            Aspect ratio to show images in. Defaults to 1.5.

    Returns:
        None. Displays an interactive widget in a Jupyter environment.
    """
    mask_set_names = list(mask_sets.keys())

    # Determine if there are multiple image types per session
    if isinstance(images[0], dict):
        multiple_img_types = True
        img_type_names = list(images[0].keys())
    else:
        multiple_img_types = False
        img_type_names = ['']

    # Setup UI widgets
    session_ui = widgets.IntSlider(min=0, max=len(sessions)-1, step=1, value=0, continuous_update=True, description='Session:')
    img_ui = widgets.Dropdown(options=img_type_names, value=img_type_names[0], description='Img Type:')
    set_ui = widgets.Dropdown(options=mask_set_names, value=mask_set_names[0], description='Mask Type:')
    opacity_ui = widgets.FloatSlider(min=0, max=1, step=0.1, value=0.5, continuous_update=True, description='Mask Opacity:')
    masks_ui = widgets.Checkbox(True, description='Show Cell Masks')
    if multiple_img_types:
        ui = HBox([VBox([img_ui, session_ui]), masks_ui, VBox([set_ui, opacity_ui])])
    else:
        ui = HBox([VBox([session_ui]), masks_ui, VBox([set_ui, opacity_ui])])

    # Colormap for cell masks
    vals = np.linspace(0, 1, 10000)
    np.random.seed(4)
    np.random.shuffle(vals)
    cmap = ListedColormap(plt.get_cmap('hsv')(vals))

    # Setup figure
    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = fig.subplots()
    ax.axis('off')
    if multiple_img_types:
        handle_main = ax.imshow(images[0][img_type_names[0]], cmap='gray', interpolation='none')
    else:
        handle_main = ax.imshow(images[0], cmap='gray', interpolation='none')
    label_mask = mask_sets[mask_set_names[0]][0] if isinstance(mask_sets[mask_set_names[0]], list) else mask_sets[mask_set_names[0]][0]
    label_mask = np.ma.masked_where(label_mask == 0, label_mask)
    handle_overlay = ax.imshow(label_mask, cmap=cmap, alpha=0.5, interpolation='none', vmin=1, vmax=20000)
    ax.set_aspect(aspect_ratio)
    plt.tight_layout()
    fig.canvas.header_visible = False
    # fig.canvas.footer_visible = False

    def update_display(
        session: int,
        img_type: str,
        mask_set: str,
        show_masks: bool,
        opacity: float
    ) -> None:
        """
        Update the displayed image and mask overlay based on widget state.

        Args:
            session (int): Index of the session to display.
            img_type (str): Image type to display (e.g., 'mean_img').
            mask_set (str): Mask set to overlay.
            show_masks (bool): Whether to show cell masks.
            opacity (float): Opacity of the mask overlay.
        """
        # Set title
        ax.set_title(f"date: {sessions[session]['date']}, session: {sessions[session]['session_id']}", fontsize=12)
        # Show image with overlay
        if multiple_img_types:
            handle_main.set_data(images[session][img_type])
        else:
            handle_main.set_data(images[session])
        if show_masks:
            label_mask = mask_sets[mask_set]
            if isinstance(label_mask, list):
                label_mask = label_mask[session]
            if isinstance(label_mask, np.ndarray) and label_mask.ndim == 3:
                label_mask = label_mask[session]
            label_mask = np.ma.masked_where(label_mask == 0, label_mask)
        else:
            label_mask = np.ma.masked_where(np.zeros((1, 1)) == 0, np.zeros((1, 1)))
        handle_overlay.set_data(label_mask)
        handle_overlay.set_alpha(opacity)
        handle_main.autoscale()

    out = widgets.interactive_output(
        update_display,
        {
            'session': session_ui,
            'img_type': img_ui,
            'mask_set': set_ui,
            'show_masks': masks_ui,
            'opacity': opacity_ui
        }
    )
    display(ui, out)
