MAX_VALUES_BY_INPUT_DTYPE = {
    "uint8": 255,
    "float32": 1.0,
}


def target_requires_grad(target):
    return target == "image_batch"
