import pathlib

UNKNOWN_LABEL = "unknown"

def get_split_and_class(path: pathlib.Path):
    if path.parent.name == "test":
        return path.parent.name, UNKNOWN_LABEL

    return path.parent.parent.name, path.parent.name
