# Copyright (c) CIGIT HPC Lab. All rights reserved
""" 通用数据加载路由 """
from __future__ import annotations
import inspect
from typing import Any, Dict

def _safe_call(fn, prefer: Dict[str, Any], extra: Dict[str, Any], *, verbose: bool = False):
    """按底层函数签名做白名单过滤并去重调用。"""
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    prefer_keys = set(prefer.keys())
    extra_keys = set(extra.keys())
    
    conflict = sorted(list(prefer_keys & extra_keys))
    invalid = sorted(list(extra_keys - allowed))
    
    filtered_extra = {k: v for k, v in extra.items() if k in allowed and k not in prefer}
    filtered_prefer = {k: v for k, v in prefer.items() if k in allowed}
    
    if verbose:
        if conflict:
            print(f"[safe_call] drop duplicated keys from extra: {conflict}")
        if invalid:
            print(f"[safe_call] drop invalid keys (not in signature): {invalid}")
    
    merged = {**filtered_extra, **filtered_prefer}
    return fn(**merged)

def load_data(
    dataname: str,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    data_root: str,
    dist: bool = False,
    **kwargs,
):
    """路由到具体数据集的 load_data 实现。"""
    # —— 统一读取公共/默认配置 ——
    cfg_dataloader = dict(
        pre_seq_length=kwargs.get('pre_seq_length', 10),
        aft_seq_length=kwargs.get('aft_seq_length', 10),
        in_shape=kwargs.get('in_shape', None),
        distributed=dist,
        use_augment=kwargs.get('use_augment', False),
        use_prefetcher=kwargs.get('use_prefetcher', False),
        drop_last=kwargs.get('drop_last', False),
    )

    if 'ThyroTriples' in dataname:
        # radom num in shape (304,304,3,10) Imitated video dataloader
        from .dataloader_thyrotriples import load_data as load_thyrotriples
        prefer = dict(
            # —— 核心参数 ——
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=val_batch_size,
            num_workers=num_workers,
            data_root=data_root,
            # —— 其他参数 ——
            in_shape=cfg_dataloader['in_shape'],
            pre_seq_length=cfg_dataloader['pre_seq_length'],
            aft_seq_length=cfg_dataloader['aft_seq_length'],
            distributed=cfg_dataloader['distributed'],
            use_augment=cfg_dataloader['use_augment'],
            use_prefetcher=cfg_dataloader['use_prefetcher'],
            drop_last=cfg_dataloader['drop_last'],
        )
        extra_kwargs = kwargs.copy()
        # 移除已经在 prefer 中设置的参数，避免重复
        for key in ['pre_seq_length', 'aft_seq_length', 'in_shape', 'distributed',
                    'use_augment', 'use_prefetcher', 'drop_last']:
            if key in extra_kwargs:
                del extra_kwargs[key]
        return _safe_call(load_thyrotriples, prefer, extra_kwargs, verbose=True)  # 开启 verbose 调试
    
        

    # elif 'rain_fall' in dataname:
    #     from .dataloader_rainfall import load_data as load_rainfall
    #     def get_region_info(region_key: str):
    #         if region_key not in region_map:
    #             raise KeyError(f"区域 {region_key} 未在 region_map 中配置，可选：{list(region_map.keys())}")
    #         return region_map[region_key]
        
    #     # 1) 时间窗口
    #     Tin = int(cfg_dataloader['pre_seq_length'])
    #     Tout = int(cfg_dataloader['aft_seq_length'])
    #     idx_in = list(range(-Tin + 1, 1))
    #     idx_out = list(range(1, Tout + 1))

    #     # 2) 计算 total_length
    #     total_length = Tin + Tout

    #     # 3) 统计文件和缓存目录 - 修正默认值
    #     stats_json = kwargs.get('stats_json', f"{data_root}/stats.json")
    #     # 从 kwargs 获取 index_cache_dir，如果没有则使用 stats_json 的目录
    #     index_cache_dir = kwargs.get('index_cache_dir', None)

    #     # 4) 区域键 - 使用更合理的默认值
    #     train_region = kwargs.get('train_region', 'TrainSet')
    #     valid_region = kwargs.get('valid_region', 'TestSet')
    #     test_region = kwargs.get('test_region', 'ValSet')
    #     train_root, train_stats = get_region_info(train_region)
    #     valid_root, valid_stats = get_region_info(valid_region)
    #     test_root, test_stats = get_region_info(test_region)

    #     # 5) 变量组合与 NWP 子集
    #     data_name = kwargs.get('data_name', 'all')
    #     nwp_vars_keep = kwargs.get('nwp_vars_keep', None)
        
    #     # 6) DataLoader / 训练标志位
    #     use_augment = bool(cfg_dataloader.get('use_augment', False))
    #     use_prefetcher = bool(cfg_dataloader.get('use_prefetcher', False))
    #     drop_last = bool(cfg_dataloader.get('drop_last', False))
    #     distributed = bool(cfg_dataloader.get('distributed', dist))
    
    #     prefer = dict(
    #         # —— 核心参数 ——
    #         batch_size=batch_size,
    #         val_batch_size=val_batch_size,
    #         test_batch_size=val_batch_size,
    #         num_workers= num_workers,
    #         data_root=data_root,

    #         input_len=Tin,
    #         target_len=Tout,
    #         idx_in=idx_in,
    #         idx_out=idx_out,

    #         # —— 新增的形状和长度参数 ——
    #         in_shape=cfg_dataloader['in_shape'],
    #         total_length=total_length,

    #         # —— DataLoader 参数 ——
    #         distributed=distributed,
    #         use_prefetcher=use_prefetcher,
    #         drop_last=drop_last,

    #         # —— 数据路径与统计 ——
    #         stats_json={
    #             "train_stats":train_stats,
    #             "valid_stats":valid_stats,
    #             "test_stats":test_stats
    #             },
    #         index_cache_dir=index_cache_dir,

    #         # —— 降雨数据集专有参数 ——
    #         train_region=train_region,
    #         valid_region=valid_region,
    #         test_region=test_region,
    #         data_name=data_name,
    #         nwp_vars_keep=nwp_vars_keep,

    #         # —— 训练增强 ——
    #         use_augment=use_augment,
    #     )

    #     # 创建要传递的额外参数，移除已经在 prefer 中设置的参数
    #     extra_kwargs = kwargs.copy()
    #     # 移除已经在 prefer 中显式设置的参数，避免重复
    #     for key in ['stats_json', 'index_cache_dir', 'train_region', 'valid_region', 
    #                'test_region', 'data_name', 'nwp_vars_keep', 'use_augment']:
    #         if key in extra_kwargs:
    #             del extra_kwargs[key]

    #     return _safe_call(load_rainfall, prefer, extra_kwargs, verbose=True)  # 开启 verbose 调试

    else:
        raise ValueError(f'Dataname {dataname} is unsupported')