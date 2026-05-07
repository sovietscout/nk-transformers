from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    compile: bool


@dataclass
class ExperimentConfig:
    policy_holdout: str
    n_irf: int
    sample_sizes: list


@dataclass
class Paths:
    cache: Path
    checkpoints: Path
    figures: Path


@dataclass
class Config:
    device: str
    training: TrainConfig
    experiment: ExperimentConfig
    paths: Paths
    _extra: dict = field(default_factory=dict)

    @classmethod
    def from_toml(cls, toml_dict: dict) -> "Config":
        toml_dict = dict(toml_dict)
        return cls(
            device=toml_dict["device"],
            training=TrainConfig(**toml_dict["training"]),
            experiment=ExperimentConfig(**toml_dict["experiment"]),
            paths=Paths(**{k: Path(v) for k, v in toml_dict["paths"].items()}),
        )

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        return self._extra[key]

    def __setitem__(self, key, value):
        if key in ("device", "training", "experiment", "paths"):
            object.__setattr__(self, key, value)
        else:
            self._extra[key] = value

    def __contains__(self, key):
        return hasattr(self, key) or key in self._extra

    def items(self):
        result = asdict(self)
        result.update(self._extra)
        return result.items()
