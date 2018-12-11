import os


def generate_name_from_path(file_path, index=0):
    head, tail = os.path.split(file_path)
    if tail:
        path_split = [tail]
    else:
        path_split = []

    a_path = head
    while True:
        head, tail = os.path.split(a_path)
        if tail:
            path_split = [tail] + path_split
            a_path = head
        else:
            break

    try:
        return path_split[index]
    except IndexError:
        return None

def generate_name_by_file_name(file_path, index=0):
    import re

    pattern = r"A(\d\d)"
    head, tail = os.path.split(file_path)

    result = re.search(pattern, tail)
    if result:
        return result.group(1)
    else:
        return None
