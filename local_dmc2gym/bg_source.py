import numpy as np
import cv2
import random
import tqdm

class ImageSource(object):

	def get_image(self):
		pass

	def reset(self):
		pass

class RandomColorSource(ImageSource):
	def __init__(self, shape):
		self.shape = shape
		self.bg = np.zeros((self.shape[0], self.shape[1], 3))
		self.reset()

	def reset(self):
		self._color = np.random.randint(0, 256, size=(3, ))
		self.bg[:, :] = self._color

	def get_iamge(self):
		return self.bg


class NoiseSource(ImageSource):
	def __init__(self, shape, strength=255):
		self.strength = strength
		self.shape = shape

	def get_image(self):
		self.bg = np.random.randn(self.shape[0], self.shape[1], 3) * self.strength
		return self.bg


class RandomImageSource(ImageSource):
	def __init__(self, shape, difficulty, date_path):
		pass

	def reset(self):
		pass

	def get_image(self):
		pass


class RandomVideoSource(ImageSource):
	def __init__(self, shape, difficulty, date_path):
		pass

	def reset(self):
		pass

	def get_image(self):
		pass