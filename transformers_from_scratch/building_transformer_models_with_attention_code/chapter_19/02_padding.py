from numpy import array
from tensorflow import math, cast, float32

def padding_mask(input):
    # Create mask which marks the zero padding values in the input by a 1
    mask = math.equal(input, 0)
    mask = cast(mask, float32)

    return mask

input = array([1, 2, 3, 4, 0, 0, 0])
print(padding_mask(input))
