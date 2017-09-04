import os


def list_files(path, relative=False):
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            if relative:
                file_path = os.path.relpath(file_path, path)
            matches.append(file_path)
    return matches
