from omegaconf import OmegaConf
from typing import Dict, Any, Optional

class BaseArg:
    """
    所有模型参数的统一基类（工业级规范）
    提供：OmegaConf 初始化、属性绑定、动态配置重写
    """
    def __init__(self, cli_config: Optional[Dict[str, Any]] = None):
        # 1. 子类必须实现：获取默认配置
        self._default_config = self.get_default_config()
        # 2. 构建 OmegaConf 对象
        self.cfg = OmegaConf.merge(self._default_config, cli_config or {})
        # 4. 绑定为实例属性（兼容原有调用方式）
        self.set_attr()

    def get_default_config(self) -> Dict[str, Any]:
        """子类必须实现：返回模型默认配置字典"""
        raise NotImplementedError("必须实现 get_default_config() 方法")

    def set_attr(self) -> None:
        """将 cfg 下的节点绑定为属性"""
        self.training = self.cfg.training
        self.model = self.cfg.model
        self.optimizer = self.cfg.optimizer

    def rewrite(self, cli_config: Dict[str, Any]) -> None:
        """运行时动态覆盖配置（命令行 / 外部传入）"""
        self.cfg = OmegaConf.merge(self.cfg, cli_config)
        self.set_attr()

    def __repr__(self) -> str:
        """打印配置（便于调试）"""
        return OmegaConf.to_yaml(self.cfg)