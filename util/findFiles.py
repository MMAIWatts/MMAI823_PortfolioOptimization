# searches a specified directory and locates all absolute file paths.
# returns a list of file paths (iterable)

import os


def findfiles(target):
    data = []

    for path, subFolders, files in os.walk(target):
        for file in files:
            filepath = os.path.join(os.path.abspath(path), file)
            data.append(filepath)
            print(filepath)

    return data
