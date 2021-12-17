import torch
import gpytorch
from botorch.generation.sampling import SamplingStrategy
from scipy.stats import multivariate_normal


class ExtendedThompsonSampling(SamplingStrategy):
    def __init__(self, model, C_model):
        super().__init__()
        self.model = model
        self.C_model = C_model

    def forward(self, X, num_samples):
        # num_samples x batch_shape x N x m
        objectives = self.model.posterior(X, observation_noise=False).rsample(sample_shape=torch.Size([num_samples]))
        with gpytorch.settings.fast_pred_var():
            constraints = []
            posterior = self.C_model.posterior(X, observation_noise=False)
            for i in range(num_samples):
                constraint_sample_i = posterior.rsample(sample_shape=torch.Size([1]))
                constraints.append(constraint_sample_i)
                
        constraints = torch.cat(constraints)
        total_violations = torch.maximum(constraints, torch.tensor(0.)).sum(dim=2)
        c_objs = objectives.squeeze() - 10**9 * total_violations.squeeze()

        idcs = torch.argmax(c_objs, dim=-1)
        return X[idcs, :]


class HybridThompsonSampling(SamplingStrategy):
    def __init__(self, model, C_model, current_best):
        super().__init__()
        self.model = model
        self.C_model = C_model
        self.current_best = current_best

    def forward(self, X, num_samples):
        objectives = self.model.posterior(X, observation_noise=False).rsample(sample_shape=torch.Size([num_samples]))

        self.C_model.eval()
        task_covar = self.C_model.covar_module.task_covar_module._eval_covar_matrix().tolist()
        mean = self.C_model(X).mean

        feasible_prob = multivariate_normal.cdf(x=(-mean).tolist(), mean=[0.] * mean.shape[1], cov=task_covar)
        feasible_prob = torch.tensor(feasible_prob, device=X.device)
        c_ei = torch.maximum(objectives.squeeze() - self.current_best, torch.tensor(0.)) * feasible_prob

        idcs = torch.argmax(c_ei, dim=-1)
        return X[idcs, :]


class FeasibleProbSampling(SamplingStrategy):
    def __init__(self, model, C_model):
        super().__init__()
        self.model = model
        self.C_model = C_model

    def forward(self, X, num_samples):
        objectives = self.model.posterior(X, observation_noise=False).rsample(sample_shape=torch.Size([num_samples]))

        self.C_model.eval()
        task_covar = self.C_model.covar_module.task_covar_module._eval_covar_matrix().tolist()
        mean = self.C_model(X).mean

        feasible_prob = multivariate_normal.cdf(x=(-mean).tolist(), mean=[0.] * mean.shape[1], cov=task_covar)
        feasible_prob = torch.tensor(feasible_prob, device=X.device)

        _, idcs = torch.topk(feasible_prob, num_samples)
        return X[idcs, :]
