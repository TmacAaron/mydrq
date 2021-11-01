import numpy as np
import cv2
import random
import tqdm
import utils
import os


def get_img_paths(difficulty, date_path):
	num_frames = utils.DIFFICULTY_NUM_VIDEOS[difficulty]
	dataset_images = sorted(os.listdir(date_path))
	image_paths = [os.path.join(date_path, subdir) for subdir in dataset_images]
	image_paths = image_paths[:num_frames]
	return image_paths


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

	def get_iamge(self, shape=None):
		self.bg = cv2.resize(self.bg, shape) if shape else self.bg
		return self.bg


class NoiseSource(ImageSource):
	def __init__(self, shape, strength=255):
		self.strength = strength
		self.shape = shape

	def get_image(self, shape=None):
		self.bg = np.random.randn(self.shape[0], self.shape[1], 3) * self.strength
		self.bg = cv2.resize(self.bg, shape) if shape else self.bg
		return self.bg


class RandomImageSource(ImageSource):
	def __init__(self, shape, difficulty, date_path):
		self.shape = shape
		self.image_paths = get_img_paths(difficulty, date_path)
		self.image_files = []
		for image_path in self.image_paths:
			image_files = sorted(os.listdir(image_path))
			[self.image_files.append(os.path.join(image_path, image_file)) for image_file in image_files]
		self.total_frames = len(self.image_files)
		self.count = 0
		self.build_bg_arr()
		self.reset()

	def build_bg_arr(self):
		self.bg_arr = np.zeros((self.total_frames, self.shape[0], self.shape[1], 3))
		for i in range(self.total_frames):
			fname = self.image_files[i]
			img = cv2.imread(fname, cv2.IMREAD_COLOR)
			self.bg_arr[i] = cv2.resize(img, (self.shape[1], self.shape[0]))

	def reset(self):
		self.idx = np.random.randint(0, self.total_frames)

	def get_image(self, shape=None):
		self.bg = self.bg_arr[self.idx]
		self.bg = cv2.resize(self.bg, shape) if shape else self.bg
		self.count += 1
		return self.bg


class RandomVideoSource(ImageSource):
	def __init__(self, shape, difficulty, date_path):
		self.shape = shape
		self.image_paths = get_img_paths(difficulty, date_path)
		self.num_path = len(self.image_paths)
		self.idx = 0
		self.reset()
		self.build_bg_arr()

	def build_bg_arr(self):
		image_path = self.image_paths[self._loc]
		self.image_files = os.listdir(image_path)
		self.bg_arr = np.zeros((len(self.image_files), self.shape[0], self.shape[1], 3))
		for i, fname in enumerate(self.image_files):
			fpath = os.path.join(image_path, fname)
			img = cv2.imread(fpath, cv2.IMREAD_COLOR)
			self.bg_arr[i] = cv2.resize(img, (self.shape[1], self.shape[0]))

	def reset(self):
		self._loc = np.random.randint(0, self.num_path)

	def get_image(self, shape=None):
		if self.idx == len(self.image_files):
			self.reset()
			self.build_bg_arr()
			self.idx = 0
		self.bg = self.bg_arr[self.idx]
		self.idx += 1
		self.bg = cv2.resize(self.bg, shape) if shape else self.bg
		return self.bg