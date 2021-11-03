import numpy as np
import cv2
import random
import tqdm
import utils
import os

DAVIS17_TRAINING_VIDEOS = [
    'bear', 'bmx-bumps', 'boat', 'boxing-fisheye', 'breakdance-flare', 'bus',
    'car-turn', 'cat-girl', 'classic-car', 'color-run', 'crossing',
    'dance-jump', 'dancing', 'disc-jockey', 'dog-agility', 'dog-gooses',
    'dogs-scale', 'drift-turn', 'drone', 'elephant', 'flamingo', 'hike',
    'hockey', 'horsejump-low', 'kid-football', 'kite-walk', 'koala',
    'lady-running', 'lindy-hop', 'longboard', 'lucia', 'mallard-fly',
    'mallard-water', 'miami-surf', 'motocross-bumps', 'motorbike', 'night-race',
    'paragliding', 'planes-water', 'rallye', 'rhino', 'rollerblade',
    'schoolgirls', 'scooter-board', 'scooter-gray', 'sheep', 'skate-park',
    'snowboard', 'soccerball', 'stroller', 'stunt', 'surf', 'swing', 'tennis',
    'tractor-sand', 'train', 'tuk-tuk', 'upside-down', 'varanus-cage', 'walking'
]
DAVIS17_VALIDATION_VIDEOS = [
    'bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel',
    'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog', 'dogs-jump',
    'drift-chicane', 'drift-straight', 'goat', 'gold-fish', 'horsejump-high',
    'india', 'judo', 'kite-surf', 'lab-coat', 'libby', 'loading', 'mbike-trick',
    'motocross-jump', 'paragliding-launch', 'parkour', 'pigs', 'scooter-black',
    'shooting', 'soapbox'
]


def get_img_paths(difficulty, date_path, train_or_val=None):
    num_frames = utils.DIFFICULTY_NUM_VIDEOS[difficulty]
    if train_or_val is None:
        dataset_images = sorted(os.listdir(date_path))
    elif train_or_val in ['trian', 'training']:
        dataset_images = DAVIS17_TRAINING_VIDEOS
    elif train_or_val in ['val', 'validation']:
        dataset_images = DAVIS17_VALIDATION_VIDEOS
    else:
        raise Exception("train_or_val %s not defined." % train_or_val)

    image_paths = [os.path.join(date_path, subdir) for subdir in dataset_images]
    if num_frames is not None:
        if num_frames > len(image_paths) or num_frames < 0:
            raise ValueError(f'`num_bakground_paths` is {num_frames} but should not be larger than the '
                             f'number of available background paths ({len(image_paths)}) and at least 0.')
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
        self._color = np.random.randint(0, 256, size=(3,))
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
    def __init__(self, shape, difficulty, date_path, train_or_val=None):
        self.shape = shape
        self.image_paths = get_img_paths(difficulty, date_path, train_or_val)
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
    def __init__(self, shape, difficulty, date_path, train_or_val=None, ground=None):
        self.ground = ground
        self.shape = shape
        self.image_paths = get_img_paths(difficulty, date_path, train_or_val)
        self.num_path = len(self.image_paths)
        self.idx = 0
        self.reset()
        self.build_bg_arr()

    def build_bg_arr(self):
        self.image_path = self.image_paths[self._loc]
        self.image_files = os.listdir(self.image_path)
        self.bg_arr = []
        for fname in self.image_files:
            fpath = os.path.join(self.image_path, fname)
            img = cv2.imread(fpath, cv2.IMREAD_COLOR)
            if self.ground == 'forground':
                mpath = fpath.replace("JPEGImages", "Annotations").replace("jpg", "png")
                mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                mask = np.logical_not(mask)
                img[mask] = 0
            self.bg_arr.append(img)

    def reset(self):
        self._loc = np.random.randint(0, self.num_path)

    def get_image(self, obs):
        if self.idx == len(self.image_files):
            self.reset()
            self.build_bg_arr()
            self.idx = 0
        self.bg = self.bg_arr[self.idx]
        self.bg = cv2.resize(self.bg, (obs.shape[1], obs.shape[0]))
        if self.ground == 'forground':
            mask = np.logical_or(self.bg[:, :, 0] > 0, self.bg[:, :, 1] > 0, self.bg[:, :, 2]> 0)
            obs[mask] = self.bg[mask]
        else:
            mask = np.logical_and((obs[:, :, 2] > obs[:, :, 1]), (obs[:, :, 2] > obs[:, :, 0]))
            obs[mask] = self.bg[mask]
        self.idx += 1
        return obs
