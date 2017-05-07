from PIL import Image
import numpy as np

class ImageCompressor:

    def __init__(self, path):
        img = Image.open(path)
        img.load()
        self.image_as_array = np.asarray( img, dtype="float32" )
