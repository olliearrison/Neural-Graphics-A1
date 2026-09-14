import torch
from PIL import Image
import numpy as np

def lerp(v0, v1, t):
    v0 + t * (v1 - v0)

def clamp(n, min, max):
    if n < min:
        return min
    elif n > max:
        return max
    else:
        return n

class NeuralImage:
    def __init__(self, path="data/bricks.png"):
        im = Image.open("data/bricks.png")
        self.A = np.asarray(im)
        self.width = self.A.shape[0]
        self.height = self.A.shape[1]

    def getPixel(self, x, y):
        x = clamp(x, 0, self.width - 1)
        y = clamp(y, 0, self.height - 1)
        return self.A[x][y]

    # u and v are in [0,1]
    def sample(self, u, v):
        x = u * float(self.width) - 0.5
        y = v * float(self.height) - 0.5

        ix = x.floor(x)
        iy = y.floor(y)

        #? looked at  code that I wrote from graphics 
        i00 = self.getPixel(x, y)
        i10 = self.getPixel(x+1, y)
        i01 = self.getPixel(x, y+1)
        i11 = self.getPixel(x+1, y+1)

        i0 = (y - iy) * (i01 - i00) + i00
        i1 = (x - ix) * (i11 - i10) + i10
        return (x - ix) * (i1 - i0) + i0



        



        

        

        


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


I = NeuralImage()
I.sample(0,0)

print(get_device())