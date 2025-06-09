from functools import wraps
from pathlib import Path
from typing import Union, List
import os
import json
from multiprocessing import Pool
from pprint import pprint

import numpy as np
import openfoamparser_mai as Ofpp

from geometry_preprocess import pressure_field_on_surface 

def h5_encode(path, end_time):
    base_path = Path(path)
    time_path = base_path / Path(end_time)

    p = Ofpp.parse_internal_field(time_path / Path('p'))

    u = Ofpp.parse_internal_field(time_path / Path('U'))

    t = Ofpp.parse_internal_field(time_path / Path('T'))

    rho = Ofpp.parse_internal_field(time_path / Path('rho'))

    cx = Ofpp.parse_internal_field(time_path / Path('Cx'))
    cy = Ofpp.parse_internal_field(time_path / Path('Cy'))
    cz = Ofpp.parse_internal_field(time_path / Path('Cz'))

    