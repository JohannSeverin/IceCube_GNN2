from tensorflow import cos, sin, stack

def angles_to_units(angles):
    azimuthal = angles[:, 0]
    zenith    = angles[:, 1]

    z         = cos(zenith)
    y         = sin(zenith) * sin(azimuthal)
    x         = sin(zenith) * cos(azimuthal)

    return stack([x, y, z], axis = 1)


