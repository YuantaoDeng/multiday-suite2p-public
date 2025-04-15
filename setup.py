from setuptools import setup

setup(
    name = 'multiday-suite2p',
    version = '0.0.1',
    description = 'Tools for registering suit2p data from multidays',
    url = 'https://github.com/Sun-Lab-NBB/multiday-suite2p-public',
    author = 'Johan Winnubst',
    author_email = 'winnubstj@janelia.hhmi.org',
    packages = ['multiday_suite2p'],
    install_requires = [
        'jupyter',
        'imageio',
        'pyyaml',
        'scanimage-tiff-reader',
        'suite2p',
        'scikit-image',
        'ipyfilechooser',
        'tqdm',
        'napari',
        'pirt @ git+https://github.com/kushaangupta/pirt',
    ],
    zip_safe = False
)
