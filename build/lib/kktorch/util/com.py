import os, copy, shutil
from typing import List, Union


__all__ =[
    "check_type",
    "check_type_list",
    "correct_dirpath",
    "convert_1d_array",
    "makedirs",
]


def check_type(instance: object, _type: Union[object, List[object]]):
    _type = [_type] if not (isinstance(_type, list) or isinstance(_type, tuple)) else _type
    is_check = [isinstance(instance, __type) for __type in _type]
    if sum(is_check) > 0:
        return True
    else:
        return False

def check_type_list(instances: List[object], _type: Union[object, List[object]], *args: Union[object, List[object]]):
    """
    Usage::
        >>> check_type_list([1,2,3,4], int)
        True
        >>> check_type_list([1,2,3,[4,5]], int, int)
        True
        >>> check_type_list([1,2,3,[4,5,6.0]], int, int)
        False
        >>> check_type_list([1,2,3,[4,5,6.0]], int, [int,float])
        True
    """
    if isinstance(instances, list) or isinstance(instances, tuple):
        for instance in instances:
            if len(args) > 0 and isinstance(instance, list):
                is_check = check_type_list(instance, *args)
            else:
                is_check = check_type(instance, _type)
            if is_check == False: return False
        return True
    else:
        return check_type(instances, _type)

def correct_dirpath(dirpath: str) -> str:
    if os.name == "nt":
        return dirpath if dirpath[-1] == "\\" else (dirpath + "\\")
    else:
        return dirpath if dirpath[-1] == "/" else (dirpath + "/")

def convert_1d_array(arrays: List[object]):
    """
    Usage::
        >>> convert_1d_array([1,2,3, [[1,1,23],2,3]])
        [1, 2, 3, 1, 1, 23, 2, 3]
    """
    import more_itertools as itr
    arrays = copy.deepcopy(arrays)
    for i, x in enumerate(arrays):
        if not (isinstance(x, list) or isinstance(x, tuple)):
            arrays[i] = [x]
    arrays = list(itr.flatten(arrays))
    i = 0
    if len(arrays) > 0:
        while(1):
            if isinstance(arrays[i], list) or isinstance(arrays[i], tuple):
                arrays = convert_1d_array(arrays)
                i = 0
            else:
                i += 1
            if len(arrays) == i:
                break
    return arrays

def makedirs(dirpath: str, exist_ok: bool = False, remake: bool = False):
    dirpath = correct_dirpath(dirpath)
    if remake and os.path.isdir(dirpath): shutil.rmtree(dirpath)
    os.makedirs(dirpath, exist_ok = exist_ok)