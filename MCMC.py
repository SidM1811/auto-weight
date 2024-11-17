import numpy as np

class MCMC:
    def __init__(self, state_manager, num_samples = 1000, num_chains = 1):
        # Transitions the state and commits the proposal
        self.state_manager = state_manager
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.samples = []

    def sample(self):
        for _ in range(self.num_samples):
            self.state_manager.propose()
            self.state_manager.acceptance_probability()
            self.state_manager.commit()
            self.samples.append(self.state_manager.summary())
        return np.array(self.samples)