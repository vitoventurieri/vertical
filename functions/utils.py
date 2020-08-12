from pathlib import Path
import os


def get_root_dir():
    return Path(__file__).parent.parent


def get_output_dir():
    return os.path.join(get_root_dir(), 'output')
