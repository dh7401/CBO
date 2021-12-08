import torch
from botorch.generation.sampling import SamplingStrategy


class WhiteningSampling(SamplingStrategy):
    def __init__(self, model, C_model):
        super().__init__()
        self.model = model
        self.C_model = C_model

    def forward(self, X, num_samples):
        return


class FeasibleProbSampling(SamplingStrategy):
    def __init__(self, model, C_model):
        super().__init__()
        self.model = model
        self.C_model = C_model

    def forward(self, X, num_samples):
        return


class HybridThompsonSampling(SamplingStrategy):
    def __init__(self, model, C_model):
        super().__init__()
        self.model = model
        self.C_model = C_model

    def forward(self, X, num_samples):
        return
