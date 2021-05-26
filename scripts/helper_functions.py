from tensorflow import cos, sin, stack, shape, concat
import numpy as np

def angles_to_units(angles):
    azimuthal = angles[:, 0]
    zenith    = angles[:, 1]

    z         = cos(zenith)
    y         = sin(zenith) * sin(azimuthal)
    x         = sin(zenith) * cos(azimuthal)

    units     = stack([x, y, z], axis = 1)

    if shape(angles)[1] > 2:
        return concat([units, angles[:, 2:]], axis = 1)
    else:
        return units

