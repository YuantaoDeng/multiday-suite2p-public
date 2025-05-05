import os
import re
import yaml
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import create_mask_img, add_overlap_info


def find_session_folders(
    main_folder: str,
    days: list[str],
    verbose: bool = False,
    suite2p_folder_name: str = "suite2p",
) -> list[str]:
    """Find session folders containing suite2p subdirectories within the main folder.

    Args:
        main_folder (str): Directory that contains all the session folders.
        days (list[str]): list of session dates to include (YYYY_MM_DD format).
        verbose (bool, optional): If True, print the number of found session folders. Defaults to False.
        suite2p_folder_name (str, optional): Name of the suite2p folder. Defaults to "suite2p".

    Returns:
        list[str]: list of found session folder paths.
    """
    sessionfolders: list[str] = []
    for day in days:
        folder = os.path.join(main_folder, day)
        sessions = [
            id
            for id in os.listdir(folder)
            if re.match(r"^\d{1,2}$", id)
            and os.path.isdir(os.path.join(folder, id, suite2p_folder_name))
        ]
        for id in sessions:
            sessionfolders.append(os.path.join(folder, id))
    if verbose:
        print(f"Found {len(sessionfolders)} session folders")
    return sessionfolders


def import_sessions(
    data_info: dict[str, Any], settings: dict[str, Any], verbose: bool = False
) -> tuple[
    list[dict[str, str]],
    list[dict[str, np.ndarray]],
    list[list[dict[str, Any]]],
    tuple[int, int],
    list[np.ndarray],
]:
    """Import session data required for multiday registration.

    Args:
        data_info (dict[str, Any]): Data info dictionary with session folder locations and selection criteria.
        settings (dict[str, Any]): Settings dictionary for cell detection and processing.
        verbose (bool, optional): If True, print progress information. Defaults to False.

    Returns:
        tuple[list[dict[str, str]], list[dict[str, np.ndarray]], list[list[dict[str, Any]]], tuple[int, int], list[np.ndarray]]:
            - sessions: list of session info dictionaries (with 'date' and 'session_id').
            - images: list of overview images per session (dicts with 'mean_img', 'enhanced_img', 'max_img').
            - cells: list of detected cell mask dictionaries per session.
            - im_size: Image size as (height, width).
            - cell_masks: list of cell mask label images per session.

    Raises:
        NameError: If no data folders are found or if requested sessions are missing.
    """
    data_paths = list(
        Path(data_info["data"]["local_processed_root"]).glob(
            "[0-9][0-9][0-9][0-9]_[0-9][0-9]_[0-9][0-9]/[0-9]"
        )
    )
    if not data_paths:
        raise NameError(
            f"Could not find any data folders in {data_info['data']['local_processed_root']}"
        )
    if data_info["data"]["session_selection"]:
        data_paths = filter_data_paths(
            data_paths, data_info["data"]["session_selection"]
        )
    short_data_paths = [
        (Path(data_path.parts[-2]) / data_path.parts[-1]).as_posix()
        for data_path in data_paths
    ]
    for filter_path in data_info["data"]["individual_sessions"]:
        if filter_path not in short_data_paths:
            raise NameError(
                f"Could not find requested individual session {filter_path} in session selection"
            )
    cells: list[list[dict[str, Any]]] = []
    cell_masks: list[np.ndarray] = []
    images: list[dict[str, np.ndarray]] = []
    sessions: list[dict[str, str]] = []
    for data_path in data_paths:
        if (Path(data_path.parts[-2]) / data_path.parts[-1]).as_posix() in data_info[
            "data"
        ]["individual_sessions"]:
            print(
                f"{Path(data_path.parts[-2]) / Path(data_path.parts[-1])}: skipping (individual session)"
            )
            continue
        combined_folder = data_path / data_info["data"]["suite2p_folder"] / "combined"
        sessions.append(
            {"date": data_path.parts[-2], "session_id": data_path.parts[-1]}
        )
        if not combined_folder.is_dir():
            raise NameError(f"Could not find combined suite2p folder for: {data_path}")
        else:
            ops = np.load(combined_folder / "ops.npy", allow_pickle=True).item()
            stat = np.load(combined_folder / "stat.npy", allow_pickle=True)
            iscell = np.load(combined_folder / "iscell.npy", allow_pickle=True)
            images.append(
                {
                    "mean_img": ops["meanImg"],
                    "enhanced_img": ops["meanImgE"],
                    "max_img": ops["max_proj"],
                }
            )
            selected_cells = [
                {
                    key: mask[key]
                    for key in ["xpix", "ypix", "lam", "med", "radius", "overlap"]
                }
                for icell, mask in enumerate(stat)
                if (iscell[icell, 1] > settings["cell_detection"]["prob_threshold"])
                and (mask["npix"] < settings["cell_detection"]["max_size"])
            ]
            filtered_cells = []
            for cell in selected_cells:
                flag = True
                for border in settings["cell_detection"]["stripe_borders"]:
                    if (
                        cell["med"][1]
                        >= (border - settings["cell_detection"]["stripe_margin"])
                    ) and (
                        cell["med"][1]
                        <= (border + settings["cell_detection"]["stripe_margin"])
                    ):
                        flag = False
                if flag:
                    filtered_cells.append(cell)
            cells.append(filtered_cells)
            if verbose:
                print(
                    f"{Path(data_path.parts[-2]) / Path(data_path.parts[-1])} contained info " \
                    f"for {len(filtered_cells)} cells"
                )
            im_size = images[0]["mean_img"].shape
            cell_masks.append(
                create_mask_img(filtered_cells, im_size, mark_overlap=True)
            )
    if data_info["data"].get("processed_data_folder"):
        base_dir = Path(data_info["data"]["processed_data_folder"]) / data_info["data"]["output_folder"]
    else:
        base_dir = Path(data_info["data"]["local_processed_root"]) / data_info["data"]["output_folder"]
    (base_dir / "registration_data").mkdir(parents=True, exist_ok=True)
    return sessions, images, cells, im_size, cell_masks


def export_masks_and_images(
    deformed_cell_masks: list[list[dict[str, Any]]],
    cell_templates: list[dict[str, Any]],
    trans_images: list[dict[str, np.ndarray]],
    images: list[dict[str, np.ndarray]],
    sessions: list[dict[str, str]],
    data_info: dict[str, Any],
    settings: dict[str, Any],
) -> None:
    """Export masks, images, and general info to the multi-day output folder.

    Args:
        deformed_cell_masks (list[list[dict[str, Any]]]): Backwards transformed cell mask info for each session.
        cell_templates (list[dict[str, Any]]): Template cell mask information.
        trans_images (list[dict[str, np.ndarray]]): Transformed images per session.
        images (list[dict[str, np.ndarray]]): Original images per session.
        sessions (list[dict[str, str]]): Session info from import_sessions.
        data_info (dict[str, Any]): Data info dictionary.
        settings (dict[str, Any]): Settings dictionary.
    """
    if data_info["data"].get("processed_data_folder"):
        output_folder = Path(data_info["data"]["processed_data_folder"]) / data_info["data"]["output_folder"]
    else:
        output_folder = Path(data_info["data"]["local_processed_root"]) / data_info["data"]["output_folder"]
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Saving multi-day info in {output_folder}.. ")
    data_paths = [
        (Path(session["date"]) / session["session_id"]).as_posix()
        for session in sessions
    ]
    info = {
        "suite2p_folder": data_info["data"]["suite2p_folder"],
        "data_paths": data_paths,
    }
    np.save(output_folder / "info.npy", info)
    np.save(output_folder / "backwards_deformed_cell_masks.npy", deformed_cell_masks)
    np.save(output_folder / "cell_templates.npy", cell_templates)
    np.save(output_folder / "trans_images.npy", trans_images)
    np.save(output_folder / "original_images.npy", images)
    np.save(output_folder / "demix_settings.npy", settings["demix"])


def import_settings_file() -> dict[str, Any]:
    """Import meta data file with path, animal info, and settings.

    Returns:
        dict[str, Any]: Contents of the meta data file.
    """
    file_loc = select_meta_file()
    with open(file_loc) as data_file:
        meta_info: dict[str, Any] = yaml.load(data_file, Loader=yaml.FullLoader)
    return meta_info


def registration_data_folder(settings: dict[str, Any]) -> Path:
    """Get the path to the registration data folder based on settings.

    Args:
        settings (dict[str, Any]): Settings dictionary containing data paths.

    Returns:
        Path: Path to the registration data folder.
    """
    if settings["data"].get("processed_data_folder"):
        return (
            Path(settings["data"]["processed_data_folder"])
            / settings["data"]["output_folder"]
            / "registration_data"
        )
    else:
        return (
            Path(settings["data"]["local_processed_root"])
            / settings["data"]["output_folder"]
            / "registration_data"
        )


def filter_data_paths(
    data_paths: list[Path], data_selection: list[list[str]]
) -> list[Path]:
    """Filter data paths according to selection filters (e.g. [['2020_01_01/0', '2020_01_10'], ['2020_01_02']]).

    Args:
        data_paths (list[Path]): Data paths to filter through.
        data_selection (list[list[str]]): Filter criteria.

    Returns:
        list[Path]: Filtered list of data paths.
    """

    def data_path_to_datetime(data_path: Path) -> pd.DatetimeIndex:
        """Helper function. Converts data paths to DateTimeIndex value.

        Session index is set as number of microseconds since start of day. 0 microseconds means no session was given.
        These values can be used for range filtering.

        Args:
            data_path (Path): Data path to convert.

        Returns:
            pd.DatetimeIndex: Datetime index for the data path.
        """
        if len(data_path.parts) == 1:
            date = data_path
            time = np.timedelta64(0, "m")
        else:
            date = data_path.parts[-2]
            time = np.timedelta64(data_path.parts[-1], "us")
        dt = pd.DatetimeIndex(
            [np.datetime64(f"{str.replace(f'{date}', '_', '-')}") + time]
        )
        return dt

    selected_data_paths: list[Path] = []
    for data_path in data_paths:
        data_path_dt = data_path_to_datetime(data_path)
        for filter_pattern in data_selection:
            filter_dt = [data_path_to_datetime(Path(f)) for f in filter_pattern]
            if len(filter_pattern) == 1:
                if filter_dt[0].microsecond == 0:
                    if filter_dt[0].date == data_path_dt.date:
                        selected_data_paths.append(data_path)
                else:
                    if filter_dt[0] == data_path_dt:
                        selected_data_paths.append(data_path)
            else:
                if filter_dt[1].microsecond == 0:
                    filter_dt[1] += np.timedelta64(10, "h")
                if (data_path_dt >= filter_dt[0]) & (data_path_dt <= filter_dt[1]):
                    selected_data_paths.append(data_path)
    return np.sort(np.unique(np.array(selected_data_paths))).tolist()


def test_extract_result_present(path: str | Path) -> bool:
    """Check whether a Suite2p output directory exists and contains expected result files.

    Args:
        path (str | Path): Path to the directory to check.

    Returns:
        bool: True if the directory exists and contains 'Fneu.npy', otherwise False.
    """
    path = Path(path)

    # Check if the path exists and is a directory
    if not path.is_dir():
        return False

    # Check if the expected result file is present in the directory
    spks_file = path / 'Fneu.npy'  # indicates the processing has completed
    return spks_file.is_file()
