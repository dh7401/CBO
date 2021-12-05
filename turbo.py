import math
from dataclasses import dataclass
from typing import List

import torch
from botorch.fit import fit_gpytorch_model
from botorch.generation.sampling import SamplingStrategy
from botorch.models import SingleTaskGP
from torch.quasirandom import SobolEngine
from botorch.models.model import Model
from torch import Tensor
from torch.nn import Module

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from mopta import mopta_evaluate

class ExtendedThompsonSampling(SamplingStrategy):
    def __init__(
        self,
        model: Model,
        C_models: List[Model],
        replacement: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.C_models = C_models
        self.replacement = replacement

    def forward(
        self, X: Tensor, num_samples: int = 1, observation_noise: bool = False
    ) -> Tensor:
        posterior = self.model.posterior(X, observation_noise=observation_noise)

        # num_samples x batch_shape x N x m
        samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
        constraints = torch.cat([C_model.posterior(X, observation_noise=observation_noise).rsample(sample_shape=torch.Size([num_samples]))
                        for C_model in C_models], dim=2)
        total_violation = torch.maximum(constraints, torch.zeros_like(constraints)).sum(dim=2)
        obj = samples.squeeze() - 10**9 * total_violation
        idcs = torch.argmax(obj, dim=-1)

        # idcs is num_samples x batch_shape, to index into X we need to permute for it
        # to have shape batch_shape x num_samples
        if idcs.ndim > 1:
            idcs = idcs.permute(*range(1, idcs.ndim), 0)
        # in order to use gather, we need to repeat the index tensor d times
        idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1))
        # now if the model is batched batch_shape will not necessarily be the
        # batch_shape of X, so we expand X to the proper shape
        Xe = X.expand(*obj.shape[1:], X.size(-1))
        # finally we can gather along the N dimension
        return torch.gather(Xe, -2, idcs)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

dim = 124
batch_size = 10
n_init = 130
n_constraints = 68
max_cholesky_size = float("inf")  # Always use Cholesky


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = float("nan")  # Note: Post-initialized
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(float(self.dim) / self.batch_size)
        self.success_tolerance = max(3, math.ceil(self.dim / 10))


def update_state(state, Y_next, C_next):
    if torch.any(torch.all(C_next <= 0, dim=1)):
        new_best = max(Y_next[torch.all(C_next <= 0, dim=1), :]).item()
    else:
        new_best = -float("inf")


    if new_best > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, new_best)
    if state.length < state.length_min:
        state.restart_triggered = True
    
    return state

state = TurboState(dim=dim, batch_size=batch_size)

def get_initial_points(dim, n_pts, seed=1):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init

def generate_batch(
    state,
    model,  # GP model for objective
    C_models, # GP models for constraints
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    C,  # Constraints
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
):
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

    if torch.any(torch.all(C <= 0, dim=1)):
        CY = torch.where(torch.all(C <= 0, dim=1, keepdim=True), Y, -float("inf"))
        x_center = X[CY.argmax(), :].clone()
    else:
        total_violation = torch.maximum(C, torch.zeros_like(C)).sum(dim=1)
        x_center = X[total_violation.argmin(), :].clone()
        print(min(total_violation))

    # Scale the TR to be proportional to the lengthscales
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    dim = X.shape[-1]
    sobol = SobolEngine(dim, scramble=True)
    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    # Create a perturbation mask
    prob_perturb = min(20.0 / dim, 1.0)
    mask = (
        torch.rand(n_candidates, dim, dtype=dtype, device=device)
        <= prob_perturb
    )
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

    # Create candidate points from the perturbations and the mask        
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    # Sample on the candidate points
    thompson_sampling = ExtendedThompsonSampling(model=model, C_models=C_models)
    with torch.no_grad():  # We don't need gradients when using TS
        X_next = thompson_sampling(X_cand, num_samples=batch_size)

    return X_next

X_turbo = get_initial_points(dim, n_init)
Y_turbo = torch.tensor(
    [-mopta_evaluate(x)[0] for x in X_turbo], dtype=dtype, device=device
).unsqueeze(-1)
C_turbo = torch.stack([mopta_evaluate(x)[1:] for x in X_turbo]).to(device)

state = TurboState(dim, batch_size=batch_size)


N_CANDIDATES = min(5000, 200 * dim)


while not state.restart_triggered:  # Run until TuRBO converges
    # Fit a GP model
    train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
    )

    model = SingleTaskGP(X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    C_models = []
    C_mlls = []
    for i in range(n_constraints):
        train_C = C_turbo[:, [i]]
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
        )
        C_models.append(SingleTaskGP(X_turbo, train_C, covar_module=covar_module, likelihood=likelihood))
        C_mlls.append(ExactMarginalLogLikelihood(C_models[-1].likelihood, C_models[-1]))

    # Do the fitting and acquisition function optimization inside the Cholesky context
    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        # Fit the model
        fit_gpytorch_model(mll)
        for i in range(n_constraints):
            fit_gpytorch_model(C_mlls[i])
    
        # Create a batch
        X_next = generate_batch(
            state=state,
            model=model,
            C_models=C_models,
            X=X_turbo,
            Y=Y_turbo,
            C=C_turbo,
            batch_size=batch_size,
            n_candidates=N_CANDIDATES,
        )

    Y_next = torch.tensor([-mopta_evaluate(x)[0] for x in X_next], dtype=dtype, device=device).unsqueeze(-1)
    C_next = torch.stack([mopta_evaluate(x)[1:] for x in X_next]).to(device)

    # Update state
    state = update_state(state=state, Y_next=Y_next, C_next=C_next)

    # Append data
    X_turbo = torch.cat((X_turbo, X_next), dim=0)
    Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)
    C_turbo = torch.cat((C_turbo, C_next), dim=0)

    # Print current status
    print(
        f"{len(X_turbo)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}"
    )
