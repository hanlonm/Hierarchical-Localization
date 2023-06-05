import numpy as np
import math
from scipy.spatial.transform import Rotation
import random
from pytransform3d import transformations as pt


def cartesian_to_spherical(x, y, z):
    theta = math.atan2(math.sqrt(x**2 + y**2), z)
    phi = math.atan2(y, x) if x >= 0 else math.atan2(y, x) + math.pi
    return theta, phi


def get_orientation_samples(num, phi_range, theta_range):
    samples = []
    while len(samples) < num:
        phi = 2 * math.pi * np.random.uniform()
        theta = math.acos(1 - 2 * np.random.uniform())

        if phi > phi_range:
            continue
        phi = phi - (phi_range / 2)

        theta = -(theta - math.pi / 2)
        if abs(theta) > theta_range / 2:
            continue

        samples.append((phi, theta))
        #print(len(samples))
    return samples


#print(get_orientation_samples(100, math.pi / 3, math.pi / 9))
