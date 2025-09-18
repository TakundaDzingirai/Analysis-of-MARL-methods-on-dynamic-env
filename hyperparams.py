from dataclasses import dataclass

@dataclass
class HyperParams:
    """Hyperparameter configuration"""
    alpha: float = 0.28457111354474063
    gamma: float = 0.99
    epsilon: float = 0.7849863661813384
    epsilon_decay: float = 0.998
    epsilon_min: float = 0.059031733973677815
    alpha_decay: float = 0.98
    alpha_min: float = 0.05
    reward_scaling: float = 1.4656855889679763
    exploration_bonus: float = 0.2

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}