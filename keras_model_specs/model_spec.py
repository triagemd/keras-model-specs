import os
import copy
import json
import keras
import numpy as np
import importlib


from six import string_types
from keras.preprocessing.image import load_img


def between_plus_minus_1(x, args=None):
    # equivalent to keras_applications.imagenet_utils.preprocess_input with mode=tf, output range [-1, 1]
    x /= 127.5
    x -= 1.
    return x


def mean_subtraction(x, args=None):
    # subtract means then normalize to between -2 and 2 (equivalent to old Keras preprocessing function)
    mean_r, mean_g, mean_b = args
    x -= [mean_r, mean_g, mean_b]
    x /= 127.5
    return x


def mean_subtraction_plus_minus_1(x, args=None):
    # subtract means then normalize to between -1 and 1
    mean_r, mean_g, mean_b = args
    x -= [mean_r, mean_g, mean_b]
    x /= 255.
    return x


def bgr_mean_subtraction(x, args=None):
    # equivalent to keras.applications.imagenet_utils.preprocess_input with mode=caffe, output range [-255, 255]
    mean_r, mean_g, mean_b = args
    x -= [mean_r, mean_g, mean_b]
    x = x[..., ::-1]
    return x


def mean_std_normalization(x, args=None):
    # equivalent to keras.applications.imagenet_utils.preprocess_input with mode=torch
    mean_r, mean_g, mean_b, std_r, std_g, std_b = args
    x -= [mean_r, mean_g, mean_b]
    x /= [std_r, std_g, std_b]
    return x


PREPROCESS_FUNCTIONS = {
    'between_plus_minus_1': between_plus_minus_1,
    'mean_subtraction': mean_subtraction,
    'bgr_mean_subtraction': bgr_mean_subtraction,
    'mean_std_normalization': mean_std_normalization
}


SPEC_FIELDS = ['name', 'model', 'target_size', 'preprocess_func', 'preprocess_args']


with open(os.path.join(os.path.split(__file__)[0], 'model_specs.json')) as file:
    BASE_SPECS = json.load(file)
    BASE_SPEC_NAMES = BASE_SPECS.keys()


class ModelSpec(object):

    @classmethod
    def get(cls, base_spec_name, **overrides):
        spec = copy.copy(BASE_SPECS.get(base_spec_name, {}))
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
        self.model = None
        self.str_model = None
        self.target_size = None
        self.preprocess_func = None
        self.preprocess_args = None
        self.keras_kwargs = {'backend': keras.backend,
                             'layers': keras.layers,
                             'models': keras.models,
                             'utils': keras.utils}

        self.__dict__.update(spec)

        self.preprocess_input = lambda x: PREPROCESS_FUNCTIONS[self.preprocess_func](x, args=self.preprocess_args)

        if isinstance(self.model, string_types):
            self.str_model = self.model
            self.model = self._get_module_class(self.model)

    def as_json(self):
        if self.str_model:
            model = self.str_model
        else:
            model = '.'.join([self.model.__module__, self.model.__name__]) if self.model else None
        return {
            'name': self.name,
            'model': model,
            'target_size': self.target_size,
            'preprocess_func': self.preprocess_func,
            'preprocess_args': self.preprocess_args,
        }

    def load_image(self, image_path):
        img = load_img(image_path, target_size=self.target_size[:2])
        image_data = np.asarray(img, dtype=np.float32)
        image_data = np.expand_dims(image_data, axis=0)
        image_data = self.preprocess_input(image_data)
        return image_data

    def _get_module_class(self, module_class_path):
        module_and_class_parts = module_class_path.split('.')
        module = importlib.import_module('.'.join(module_and_class_parts[:-1]))
        return getattr(module, module_and_class_parts[-1])
