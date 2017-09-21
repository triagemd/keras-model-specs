from keras.applications.mobilenet import MobileNet

from keras_model_specs import ModelSpec


EXPECTED_BASE_SPECS = ['inception_v3', 'mobilenet_v1', 'xception', 'resnet50', 'vgg16', 'vgg19']


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
