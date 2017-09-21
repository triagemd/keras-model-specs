import pytest
import os

from keras.applications.mobilenet import MobileNet
from keras.optimizers import SGD

from keras_model_specs import ModelSpec


EXPECTED_BASE_SPECS = [
    'inception_v3',
    'inception_v4',
    'mobilenet_v1',
    'xception',
    'resnet50',
    'resnet152',
    'vgg16',
    'vgg19'
]


def assert_model_predict(spec_name, expected_classes):
    spec = ModelSpec.get(spec_name, preprocess_args=[1, 2, 3])
    model = spec.klass()
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    image_data = spec.load_image('tests/files/cat.jpg')
    out = model.predict(image_data)
    assert len(out.tolist()[0]) == expected_classes


def test_has_all_base_specs():
    for name in EXPECTED_BASE_SPECS:
        spec = ModelSpec.get(name)
        assert spec is not None


def test_returns_none_for_nonexistent_and_spec():
    spec = ModelSpec.get('nonexistent_v1')
    assert spec is None


def test_returns_nonexistent_with_overrides():
    spec = ModelSpec.get(
        'nonexistent_v1',
        klass='keras.applications.mobilenet.MobileNet',
        target_size=[224, 224, 3],
        preprocess_func='mean_subtraction',
        preprocess_args=[1, 2, 3]
    )
    assert spec is not None
    assert spec.klass == MobileNet
    assert spec.target_size == [224, 224, 3]
    assert spec.preprocess_func == 'mean_subtraction'
    assert spec.preprocess_args == [1, 2, 3]
    assert spec.preprocess_input is not None


def test_returns_existing_with_overrides():
    spec = ModelSpec.get(
        'mobilenet_v1',
        klass='keras.applications.mobilenet.MobileNet',
        target_size=[512, 512, 3],
        preprocess_func='mean_subtraction',
        preprocess_args=[1, 2, 3]
    )
    assert spec is not None
    assert spec.klass == MobileNet
    assert spec.target_size == [512, 512, 3]
    assert spec.preprocess_func == 'mean_subtraction'
    assert spec.preprocess_args == [1, 2, 3]
    assert spec.preprocess_input is not None


def test_load_image_for_all_base_specs():
    for name in EXPECTED_BASE_SPECS:
        spec = ModelSpec.get(name, preprocess_args=[1, 2, 3])
        image_data = spec.load_image('tests/files/cat.jpg')
        assert image_data.any()


def test_model_inception_v3():
    assert_model_predict('inception_v3', 1000)


def test_model_inception_v4():
    assert_model_predict('inception_v4', 1001)


def test_model_mobilenet_v1():
    assert_model_predict('mobilenet_v1', 1000)


def test_model_resnet50():
    assert_model_predict('resnet50', 1000)


def test_model_resnet152():
    assert_model_predict('resnet152', 1000)


def test_model_vgg16():
    assert_model_predict('vgg16', 1000)


@pytest.mark.skipif('CI' in os.environ, reason='requires too much memory')
def test_model_vgg19():
    assert_model_predict('vgg19', 1000)


def test_model_xception():
    assert_model_predict('xception', 1000)
