# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ctypes
import torch
c_float = ctypes.c_float
c_float_p = ctypes.POINTER(ctypes.c_float)
c_float_p_p = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
c_int = ctypes.c_int
c_int_p = ctypes.POINTER(ctypes.c_int)
c_int_p_p = ctypes.POINTER(ctypes.POINTER(ctypes.c_int))
c_int64 = ctypes.c_int64
c_int64_p = ctypes.POINTER(ctypes.c_int64)


def tensor_ptr(tensor):
    tensor_addr = tensor.storage().data_ptr()
    tensor_ptr = None
    if tensor.dtype is torch.float32:
        tensor_ptr = ctypes.cast(tensor_addr, c_float_p)
    elif tensor.dtype is torch.int32:
        tensor_ptr = ctypes.cast(tensor_addr, c_int_p)
    elif tensor.dtype is torch.int64:
        tensor_ptr = ctypes.cast(tensor_addr, c_int64_p)
    else:
        raise Exception(f"Dtype is not suppported: {tensor.dtype}")
    return tensor_ptr


def deduce_signatrue(tensors):
    sig = []
    for p in tensors:
        if p.dtype is torch.float32:
            sig.append(c_float_p)
        elif p.dtype is torch.int32:
            sig.append(c_int_p)
        elif p.dtype is torch.int64:
            sig.append(c_int64_p)
        else:
            raise Exception(f"Dtype is not suppported: {p.dtype}")
    return tuple(sig)


def get_data_addr(tensors):
    addr = [tensor_ptr(p) for p in tensors]
    return tuple(addr)
