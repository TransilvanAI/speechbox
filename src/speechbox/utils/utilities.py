import datetime
from typing import List
import torch

def seconds_to_srt_timestamp(seconds):
    """Convert seconds to SRT timestamp format."""
    ms = int((seconds - int(seconds)) * 1000)
    td = datetime.timedelta(seconds=int(seconds))
    time = (datetime.datetime(1, 1, 1) + td).strftime('%H:%M:%S')
    return f"{time},{ms:03d}"


def get_device(device: str = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if (mps := getattr(torch.backends, 'mps', None)) is not None and mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def batch(x: List, bs:int=32):
    l = len(x)
    min_i = 0
    max_i = min(bs, l)
    while min_i < l:
        yield x[min_i:max_i]
        min_i = min_i + bs
        max_i = min(max_i + bs, l)
