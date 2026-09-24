from dataclasses import dataclass


@dataclass
class ModelConfig:
    # 序列编码器相关超参数
    emb_dim: int = 128  # 输入 POI 嵌入维度
    n_heads: int = 4  # 多头注意力头数
    ff_hidden: int = 256  # 前馈网络隐藏维度
    dropout: float = 0.1  # Dropout 比例

    # 滑动窗口与聚类相关超参数
    window_size: int = 5  # 滑动窗口大小
    guidance_w: float = 0.3  # 负向引导权重 w
    dbscan_eps: float = 0.5  # DBSCAN 半径超参数
    dbscan_min_samples: int = 2  # DBSCAN 最小样本数

    # 其他
    device: str = "cuda"  # 设备设定，可为 "cuda" 或 "cpu"


def get_default_config() -> ModelConfig:
    return ModelConfig()

