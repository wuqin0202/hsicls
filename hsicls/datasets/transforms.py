import numpy as np

class RadiationNoise:
    def __init__(self, alpha_range=(0.9, 1.1), beta=1 / 25):
        self.alpha_range = alpha_range
        self.beta = beta

    def __call__(self, image):
        alpha = np.random.uniform(*self.alpha_range).astype(image.dtype)
        noise = np.random.normal(loc=0.0, scale=1.0, size=image.shape).astype(image.dtype)
        return alpha * image + self.beta * noise

class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        horizontal = np.random.random() > self.p
        vertical = np.random.random() > self.p
        if horizontal:
            image = np.fliplr(image)
        if vertical:
            image = np.flipud(image)
        return image