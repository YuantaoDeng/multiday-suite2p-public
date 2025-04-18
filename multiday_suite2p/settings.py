from getpass import getpass
from pathlib import Path
from typing import Any

import yaml
from ipyfilechooser import FileChooser
from IPython.display import display


def select_settings_file(start_dir: str = '../') -> FileChooser:
    """Launch a file chooser dialog to select a YAML settings file.

    Args:
        start_dir (str, optional): Directory to start the file chooser in. Defaults to '../'.

    Returns:
        FileChooser: An ipyfilechooser FileChooser object for user interaction.
    """
    fc = FileChooser(start_dir)
    fc.use_dir_icons = True
    fc.filter_pattern = '*.yml'
    display(fc)
    return fc


def parse_settings(file: str, request_pass: bool = False) -> dict[str, Any]:
    """Parse a YAML settings file and validate required keys. Optionally request a server password.

    Args:
        file (str): Path to the YAML settings file.
        request_pass (bool, optional): If True, prompt user for server password. Defaults to False.

    Returns:
        dict[str, Any]: Parsed settings dictionary with required keys.

    Raises:
        NameError: If any required key is missing in the settings file.
    """
    file_path = Path(file)
    if file_path.is_file():
        with open(file_path) as data_file:
            settings: dict[str, Any] = yaml.load(data_file, Loader=yaml.FullLoader)

        required_keys = ['server', 'cell_detection', 'registration', 'clustering', 'demix']
        for key in required_keys:
            if key not in settings:
                raise NameError(f"Could not find key '{key}' in settings file")
        if request_pass:
            settings['server']['password'] = getpass('Enter your server password: ')
        return settings
    else:
        raise FileNotFoundError(f"Settings file not found: {file}")


def parse_data_info(file: str) -> dict[str, Any]:
    """Parse a YAML data info file and validate required keys. Ensures server paths are set.

    Args:
        file (str): Path to the YAML data info file.

    Returns:
        dict[str, Any]: Parsed data info dictionary with required keys and server paths set.

    Raises:
        NameError: If any required key is missing in the data info file.
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(file)
    if file_path.is_file():
        with open(file_path) as data_file:
            settings: dict[str, Any] = yaml.load(data_file, Loader=yaml.FullLoader)

        required_keys = ['data', 'animal']
        for key in required_keys:
            if key not in settings:
                raise NameError(f"Could not find key '{key}' in settings file")
        # Set server paths if not provided
        if not settings['data'].get('server_processed_root'):
            settings['data']['server_processed_root'] = settings['data']['local_processed_root']
        if not settings['data'].get('server_bin_root'):
            settings['data']['server_bin_root'] = settings['data']['local_bin_root']
        return settings
    else:
        raise FileNotFoundError(f"Data info file not found: {file}")
