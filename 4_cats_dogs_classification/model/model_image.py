import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np

class ModelImage:
    def __init__(self, image_size=256):
        self.image_size = image_size
        
    def load_image(self, image_path):
        """
        Loads an image, converts to grayscale, resizes, normalizes to [0, 1],
        and returns it as a flattened float32 array.
        """
        image = plt.imread(image_path)

        # Remove alpha if present (RGBA → RGB)
        if image.shape[-1] == 4:
            image = image[..., :3]

        # Convert to grayscale (if RGB)
        if image.ndim == 3 and image.shape[-1] == 3:
            image = rgb2gray(image)

        image_size = (self.image_size, self.image_size)
        image = resize(image, image_size, anti_aliasing=True)
        image = image.astype(np.float32)

        # plt.imshow(image, cmap='gray')
        image = image.flatten()

        return image