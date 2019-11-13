A helper package for managing Keras model base architectures with overrides for target size and preprocessing functions.

[![Build Status](https://travis-ci.org/triagemd/keras-model-specs.svg?branch=master)](https://travis-ci.org/triagemd/keras-model-specs)
[![PyPI version](https://badge.fury.io/py/keras-model-specs.svg)](https://badge.fury.io/py/keras-model-specs)
[![codecov](https://codecov.io/gh/triagemd/keras-model-specs/branch/master/graph/badge.svg)](https://codecov.io/gh/triagemd/keras-model-specs)


## Code usage

Install the package through pip. The latest version expects to use tf2.0.

```
pip install keras-model-specs
```

To use use it with tf1.x install by:
```
pip install keras-model-specs==1.2.0
```

Example

```
from keras_model_specs import ModelSpec

# Select a model architecture
model_architecture = 'mobilenet_v2'
model_spec = ModelSpec.get(model_architecture)

# Model Spec Attributes
print(model_spec.preprocess_func)
print(model_spec.target_size)

# Load, resize and pre-process the image 
image_data = model_spec.load_image('tests/files/cat.jpg')
```
