from pathlib import Path


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
