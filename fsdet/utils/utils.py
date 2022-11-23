import torch


def move_to_cpu(obj):
    if (isinstance(obj, torch.Tensor)):
        obj = obj.cpu()
    elif (isinstance(obj, dict)):
        for key, value in obj.items():
            obj[key] = move_to_cpu(value)
    return obj
