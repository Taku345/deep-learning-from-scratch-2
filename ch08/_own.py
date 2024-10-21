import sys

sys.path.append("..")
from common.layers import Softmax
import numpy as np


class AttentionWeight:

    def __init__(self) -> None:
        self.params, self


N, T, H = 10, 5, 4
hs = np.random.randn(N, T, H)
h = np.random.randn(N, H)
hr = h.reshape(N, 1, H).repeat(T, axis=1)

t = hs * hr
s = np.sum(t, axis=2)
softmax = Softmax()
a = softmax.forward(s)

self.cache = (hs, hr)
