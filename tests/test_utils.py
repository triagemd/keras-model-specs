import os

from keras_model_specs.utils import list_files


def touch(path):
    with open(path, 'a'):
        os.utime(path, None)


def test_list_files(tmpdir):
    touch(os.path.join(tmpdir, 'foo.jpg'))
    os.makedirs(os.path.join(tmpdir, 'bar'))
    touch(os.path.join(tmpdir, 'bar', 'baz-1.jpg'))
    touch(os.path.join(tmpdir, 'bar', 'baz-2.jpg'))
    actual = list_files(tmpdir)
    actual = [file.replace(str(tmpdir) + '/', '') for file in actual]
    expected = ['foo.jpg', 'bar/baz-1.jpg', 'bar/baz-2.jpg']
    assert actual == expected


def test_list_files_relative(tmpdir):
    touch(os.path.join(tmpdir, 'foo.jpg'))
    os.makedirs(os.path.join(tmpdir, 'bar'))
    touch(os.path.join(tmpdir, 'bar', 'baz-1.jpg'))
    touch(os.path.join(tmpdir, 'bar', 'baz-2.jpg'))
    actual = list_files(tmpdir, relative=True)
    expected = ['foo.jpg', 'bar/baz-1.jpg', 'bar/baz-2.jpg']
    assert actual == expected
