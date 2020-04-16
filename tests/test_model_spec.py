import keras
import keras_model_specs.model_spec as model_spec


from keras_model_specs import ModelSpec
from keras_applications.mobilenet import MobileNet


def assert_lists_same_items(list1, list2):
    assert sorted(list1) == sorted(list2)


def test_has_all_base_specs():
    assert_lists_same_items(model_spec.BASE_SPEC_NAMES, model_spec.BASE_SPECS.keys())

    for name in model_spec.BASE_SPEC_NAMES:
        spec = ModelSpec.get(name)
        assert spec is not None
        assert spec.name == name
        assert spec.model is not None
        assert spec.target_size is not None
        assert spec.preprocess_func is not None
        assert spec.preprocess_input is not None
        assert spec.keras_kwargs == {'backend': keras.backend,
                                     'layers': keras.layers,
                                     'models': keras.models,
                                     'utils': keras.utils}


def test_as_json_mobilenet_v1():
    spec = ModelSpec.get('mobilenet_v1')
    expected = {
        'name': 'mobilenet_v1',
        'model': 'keras_applications.mobilenet.MobileNet',
        'preprocess_args': None,
        'preprocess_func': 'between_plus_minus_1',
        'target_size': [224, 224, 3]
    }
    assert spec.as_json() == expected


def test_returns_none_for_nonexistent_and_spec():
    spec = ModelSpec.get('nonexistent_v1')
    assert spec is None


def test_returns_nonexistent_with_overrides():
    spec = ModelSpec.get(
        'nonexistent_v1',
        model='keras_applications.mobilenet.MobileNet',
        target_size=[224, 224, 3],
        preprocess_func='mean_subtraction',
        preprocess_args=[1, 2, 3]
    )
    assert spec is not None
    assert spec.model == MobileNet
    assert spec.target_size == [224, 224, 3]
    assert spec.preprocess_func == 'mean_subtraction'
    assert spec.preprocess_args == [1, 2, 3]
    assert spec.preprocess_input is not None
    assert spec.keras_kwargs == {'backend': keras.backend,
                                 'layers': keras.layers,
                                 'models': keras.models,
                                 'utils': keras.utils}


def test_returns_existing_with_overrides():
    spec = ModelSpec.get(
        'mobilenet_v1',
        model='keras_applications.mobilenet.MobileNet',
        target_size=[512, 512, 3],
        preprocess_func='mean_subtraction',
        preprocess_args=[1, 2, 3]
    )
    assert spec is not None
    assert spec.model == MobileNet
    assert spec.target_size == [512, 512, 3]
    assert spec.preprocess_func == 'mean_subtraction'
    assert spec.preprocess_args == [1, 2, 3]
    assert spec.preprocess_input is not None
    assert spec.keras_kwargs == {'backend': keras.backend,
                                 'layers': keras.layers,
                                 'models': keras.models,
                                 'utils': keras.utils}


def test_load_image_for_all_base_specs():
    for name in model_spec.BASE_SPEC_NAMES:
        spec = ModelSpec.get(name)
        image_data = spec.load_image('tests/files/cat.jpg')
        assert image_data.any()
