import os
from pathlib import Path


def file_tree_show(location):  # noqa: E501
    """get file tree from current dir

     # noqa: E501

    :param location: Name of project to add
    :type location: str

    :rtype: FileTree
    """

    return os.listdir(location)


if __name__ == '__main__':
    print(file_tree_show('/data'))