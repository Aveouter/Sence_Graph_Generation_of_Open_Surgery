from typing import Dict, Tuple, List
from pathlib import Path
import re
import pandas as pd

dataset_parameters = {
    'ThyroTriples': {
        'in_shape': (3, 16, 64, 64),      # (C, T, H, W)
        # 'out_shape' : (3, 8, 64, 64),
        'pre_seq_length': 16,             # 输入：过去16帧
        'aft_seq_length': 1,              # 输出：预测一个场景图
        'metrics': ['mse'],        
    },
}