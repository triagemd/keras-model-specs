import os
import json
import numpy as np

from six import string_types
from keras.preprocessing.image import load_img


def between_plus_minus_1(x, args):
    # equivalent to keras.applications.mobilenet.preprocess_input
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def mean_subtraction(x, args):
    # equivalent to keras.applications.imagenet_utils.preprocess_input (with channels_first)
    mean_r, mean_g, mean_b = args
    x -= [mean_r, mean_g, mean_b]
    x /= 255.
    x *= 2.
    return x


PREPROCESS_FUNCTIONS = {
    'between_plus_minus_1': between_plus_minus_1,
    'mean_subtraction': mean_subtraction,
}


class ModelSpec(object):

    @classmethod
    def get(cls, base_spec_name, **overrides):
        with open(os.path.join(os.path.split(__file__)[0], 'model_specs.json')) as file:
            base_specs = json.load(file)
        spec = base_specs.get(base_spec_name, {})
        if len(spec) == 0 and len(overrides) == 0:
            return None
        spec['name'] = base_spec_name
        spec.update(overrides)
        return ModelSpec(spec)

    def __init__(self, spec):
        self.name = None
        self.klass = None
        self.target_size = None
        self.preprocess_func = None
        self.preprocess_args = None

        self.__dict__.update(spec)

        if isinstance(self.klass, str):
            self.klass = None

        if isinstance(self.target_size, string_types):
            self.target_size = self.target_size.split(',')
        if self.target_size:
            self.target_size = [int(v) for v in self.target_size]

        if isinstance(self.preprocess_args, str):
            self.preprocess_args = self.preprocess_args.split(',')
        if self.preprocess_args:
            self.preprocess_args = [int(v) for v in self.preprocess_args]

    def load_image(self, image_path):
        preprocess_input = PREPROCESS_FUNCTIONS[self.preprocess_func]
        img = load_img(image_path, target_size=self.target_size[:2])
        image_data = np.asarray(img, dtype=np.float32)
        image_data = np.expand_dims(image_data, axis=0)
        image_data = preprocess_input(image_data, self.preprocess_args)
        return image_data
