import torch.nn as nn


LOSS_FACTORY = {
    "ce": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
}

def loss_construction(loss_name="ce"):
    try:
        return LOSS_FACTORY[loss_name]()  # 调用构造函数
    except KeyError:
        raise ValueError(f"Unknown loss type: {loss_name}")