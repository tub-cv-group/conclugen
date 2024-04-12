import torch


def sanitize_batch_data(item):
    if torch.is_tensor(item):
        return item.detach().cpu()
    elif isinstance(item, dict):
        return {k: sanitize_batch_data(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [sanitize_batch_data(v) for v in item]
    elif isinstance(item, tuple):
        return tuple(sanitize_batch_data(v) for v in item)
    else:
        raise ValueError(f'Unknown type {type(item)}')
