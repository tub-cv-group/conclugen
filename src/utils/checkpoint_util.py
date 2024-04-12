import os
from typing import List
import torch


def sanitize_ckpt(path: str, replace: dict, remove: List[str]) -> None:
    assert os.path.exists(path)
    print(f'Sanitizing checkpoint at path {path} by replacing keys {replace} and deleting {remove}.')
    ckpt = torch.load(path)
    if replace is not None and len(replace) > 0:
        new_state_dict = {k.replace(replace[0], replace[1]): v for (k, v) in ckpt['state_dict'].items()}
    else:
        new_state_dict = ckpt['state_dict']
    if remove is not None and len(remove) > 0:
        new_state_dict = {k: v for (k, v) in new_state_dict.items() if k not in remove}
    if replace is None and remove is None:  
        new_state_dict = ckpt['state_dict']
    ckpt['state_dict'] = new_state_dict
    torch.save(ckpt, path.replace('.ckpt', '_sanitized.ckpt'))
