from __future__ import annotations

from typing import Optional

from .envs.alfworld_adapter import build_alfworld_env
from .envs.base import BaseEnv
from .envs.webshop_adapter import build_webshop_env


def build_env(
    name: str,
    mode: str,
    num_products: Optional[int] = None,
    alfworld_config_path: Optional[str] = None,
    alfworld_data_dir: Optional[str] = None,
    webshop_root: Optional[str] = None,
) -> BaseEnv:
    # 这里是环境分发入口：配置里的 env.name 最终会落到具体 benchmark adapter。
    # 如果依赖没装好，不要悄悄回退成 toy env，而是明确报错，避免实验被假数据污染。
    if name == "alfworld":
        return build_alfworld_env(
            mode=mode,
            config_path=alfworld_config_path,
            data_dir=alfworld_data_dir,
        )
    if name == "webshop":
        return build_webshop_env(
            mode=mode,
            num_products=num_products or 1000,
            webshop_root=webshop_root,
        )
    raise ValueError(f"Unsupported env: {name}")
