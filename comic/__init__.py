import os
import pathlib

COMIC_DIR = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = COMIC_DIR.parent / "data"
