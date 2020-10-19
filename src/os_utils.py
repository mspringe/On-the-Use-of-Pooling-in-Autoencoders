"""
This module provides helper method for os operations.
"""
import os


def join(path):
    """
    Takes a Unix/ Linux path and maps os.path.join on it for cross platform compatability

    :param path: Unix/ Linux path
    :return: OS specific path
    """
    path = path.strip()
    if not '/' in path:
        return path
    if path[0] == '/':
        return os.path.join(os.sep, *path.split('/'))
    return os.path.join(*path.split('/'))

