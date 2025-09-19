class HyperParams:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.9, epsilon_decay=0.995,
                 epsilon_min=0.05, alpha_decay=0.95, reward_scaling=1.0,
                 exploration_bonus=0.2, lr_actor=0.001, lr_critic=0.002, tau=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alpha_decay = alpha_decay
        self.reward_scaling = reward_scaling
        self.exploration_bonus = exploration_bonus
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.tau = tau

    def to_dict(self):
        return {
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'alpha_decay': self.alpha_decay,
            'reward_scaling': self.reward_scaling,
            'exploration_bonus': self.exploration_bonus,
            'lr_actor': self.lr_actor,
            'lr_critic': self.lr_critic,
            'tau': self.tau
        }