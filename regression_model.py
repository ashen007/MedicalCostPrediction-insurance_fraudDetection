import numpy as np


def regression(x,y):
    c,i = np.linalg.lstsq(x,y,rcond=None)[0]
    return c,i

