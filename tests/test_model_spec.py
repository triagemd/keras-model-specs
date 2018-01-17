import pytest
import os

from keras.applications.mobilenet import MobileNet
from keras.optimizers import SGD

from keras_model_specs import ModelSpec
import keras_model_specs.model_spec as model_spec


EXPECTED_BASE_SPECS = [
    'densenet_121',
    'densenet_169',
    'densenet_201',
    'inception_resnet_v2',
    'inception_v3',
    'inception_v4',
    'mobilenet_v1',
    'nasnet_large',
    'nasnet_mobile',
    'xception',
    'resnet50',
    'resnet152',
    'vgg16',
    'vgg19'
]


def assert_lists_same_items(list1, list2):
    assert sorted(list1) == sorted(list2)


def assert_model_predict(spec_name, num_expected_classes=1000):
    spec = ModelSpec.get(spec_name, preprocess_args=[1, 2, 3])
    model = spec.klass()
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    image_data = spec.load_image('tests/files/cat.jpg')
    out = model.predict(image_data)
    assert len(out.tolist()[0]) == num_expected_classes


def test_has_all_base_specs():
    assert_lists_same_items(model_spec.BASE_SPECS.keys(), EXPECTED_BASE_SPECS)
    assert_lists_same_items(model_spec.BASE_SPEC_NAMES, EXPECTED_BASE_SPECS)

    for name in EXPECTED_BASE_SPECS:
        spec = ModelSpec.get(name)
        assert spec is not None
        assert spec.name == name


def test_as_json_mobilenet_v1():
    spec = ModelSpec.get('mobilenet_v1')
    expected = {
        'name': 'mobilenet_v1',
        'klass': 'keras.applications.mobilenet.MobileNet',
        'preprocess_args': None,
        'preprocess_func': 'between_plus_minus_1',
        'target_size': [224, 224, 3]
    }
    assert spec.as_json() == expected


def test_as_json_resnet50():
    spec = ModelSpec.get('resnet50')
    expected = {
        'name': 'resnet50',
        'klass': 'keras.applications.resnet50.ResNet50',
        'preprocess_args': [103.939, 116.779, 123.68],
        'preprocess_func': 'mean_subtraction',
        'target_size': [224, 224, 3]
    }
    assert spec.as_json() == expected


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


def test_model_densenet_121():
    assert_model_predict('densenet_121')


def test_model_densenet_169():
    assert_model_predict('densenet_169')


def test_model_densenet_201():
    assert_model_predict('densenet_201')


def test_model_inception_resnet_v2():
    assert_model_predict('inception_resnet_v2')


def test_model_inception_v3():
    assert_model_predict('inception_v3')


def test_model_inception_v4():
    assert_model_predict('inception_v4', 1001)


def test_model_mobilenet_v1():
    assert_model_predict('mobilenet_v1')


@pytest.mark.skipif('CI' in os.environ, reason='requires too much memory')
def test_model_nasnet_large():
    assert_model_predict('nasnet_large')


@pytest.mark.skipif('CI' in os.environ, reason='requires too much memory')
def test_model_nasnet_mobile():
    assert_model_predict('nasnet_mobile')


def test_model_resnet50():
    assert_model_predict('resnet50')


@pytest.mark.skipif('CI' in os.environ, reason='requires too much memory')
def test_model_resnet152():
    assert_model_predict('resnet152')


@pytest.mark.skipif('CI' in os.environ, reason='requires too much memory')
def test_model_vgg16():
    assert_model_predict('vgg16')


@pytest.mark.skipif('CI' in os.environ, reason='requires too much memory')
def test_model_vgg19():
    assert_model_predict('vgg19')


def test_model_xception():
    assert_model_predict('xception')
