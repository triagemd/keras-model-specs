import os
import json
import numpy as np
import importlib

from six import string_types
from keras.preprocessing.image import load_img


def between_plus_minus_1(x, args=None):
    # equivalent to keras.applications.mobilenet.preprocess_input
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def mean_subtraction(x, args=None):
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


SPEC_FIELDS = ['name', 'klass', 'target_size', 'preprocess_func', 'preprocess_args']


class ModelSpec(object):

    @classmethod
    def get(cls, base_spec_name, **overrides):
        with open(os.path.join(os.path.split(__file__)[0], 'model_specs.json')) as file:
            base_specs = json.load(file)
        spec = base_specs.get(base_spec_name, {})
        if len(spec) == 0 and len(overrides) == 0:
            return None
        spec['name'] = base_spec_name
        for field in SPEC_FIELDS:
            # Ignore incoming None fields
            if overrides.get(field) is not None:
                spec[field] = overrides[field]
        return ModelSpec(spec)

    def __init__(self, spec):
        self.name = None
        self.klass = None
        self.target_size = None
        self.preprocess_func = None
        self.preprocess_args = None

        self.__dict__.update(spec)

        self.preprocess_input = PREPROCESS_FUNCTIONS[self.preprocess_func]

        if isinstance(self.klass, string_types):
            self.klass = self._get_module_class(self.klass)

    def load_image(self, image_path):
        img = load_img(image_path, target_size=self.target_size[:2])
        image_data = np.asarray(img, dtype=np.float32)
        image_data = np.expand_dims(image_data, axis=0)
        image_data = self.preprocess_input(image_data, self.preprocess_args)
        return image_data

    def _get_module_class(self, module_class_path):
        module_and_class_parts = module_class_path.split('.')
        module = importlib.import_module('.'.join(module_and_class_parts[:-1]))
        return getattr(module, module_and_class_parts[-1])
