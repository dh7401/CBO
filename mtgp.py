import argparse
import math
from dataclasses import dataclass

import torch
from torch.quasirandom import SobolEngine
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.optim.fit import fit_gpytorch_torch
from botorch.models import SingleTaskGP, KroneckerMultiTaskGP

from mopta import mopta_evaluate
from sampling import ExtendedThompsonSampling, FeasibleProbSampling, HybridThompsonSampling


# Choosing search method
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--search", help='''Choose a search method.\n
                                        ets: Extended Thompson Sampling\n
                                        hts: Hybrid Thompson Sampling\n
                                        fprob: Feasible Probability Sampling'''
                    , choices=["ets", "hts", "fprob"], required=True)
args = parser.parse_args()


torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
dtype = torch.double
dim = 124
batch_size = 15
n_init = 50
n_constraints = 30
max_cholesky_size = float("inf")
max_queries = 500


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
    if torch.any(torch.all(C_next <= 0, dim=0)):
        new_best = max(Y_next[torch.all(C_next <= 0, dim=0)]).item()
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

def get_initial_points(dim, n_pts):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=torch.randint(100, (1,)).item())
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init

def generate_batch(
    state,
    model,  # GP model for objective
    C_model, # GP models for constraints
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    C,  # Constraints
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
):
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

    if torch.any(torch.all(C <= 0, dim=0)):
        CY = torch.where(torch.all(C <= 0, dim=0), Y, -float("inf"))
        x_center = X[CY.argmax(), :].clone()
    else:
        total_violations = torch.maximum(C, torch.tensor(0., device=device)).sum(dim=0)
        x_center = X[total_violations.argmin(), :].clone()
        print("Minimum total violation:", min(total_violations).item())

    if state.best_value == -float("inf"):
        current_best = 0.
    else:
        current_best = state.best_value

    # Scale the TR to be proportional to the lengthscales
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    dim = X.shape[-1]
    sobol = SobolEngine(dim, scramble=True, seed=torch.randint(100, (1,)).item())
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
    if args.search == "ets":
        sampling = ExtendedThompsonSampling(model=model, C_model=C_model)
    elif args.search == "hts":
        sampling = HybridThompsonSampling(model=model, C_model=C_model, current_best=current_best)
    elif args.search == "fprob":
        sampling = FeasibleProbSampling(model=model, C_model=C_model)
    with torch.no_grad():  # We don't need gradients when using TS
        X_next = sampling(X_cand, num_samples=batch_size)

    return X_next

X_turbo = get_initial_points(dim, n_init)
Y_turbo = torch.tensor([-mopta_evaluate(x)[0] for x in X_turbo], dtype=dtype, device=device).unsqueeze(-1)
C_turbo = torch.stack([mopta_evaluate(x)[1 : n_constraints + 1] for x in X_turbo], dim=1).to(device).unsqueeze(-1)

state = TurboState(dim, batch_size=batch_size)

N_CANDIDATES = min(2000, 200 * dim)


while len(X_turbo) < max_queries:
    # Fit a GP model
    train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)))
    model = SingleTaskGP(X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    C_model = KroneckerMultiTaskGP(X_turbo, C_turbo.squeeze().t().contiguous())
    C_mll = ExactMarginalLogLikelihood(C_model.likelihood, C_model)

    # Do the fitting and acquisition function optimization inside the Cholesky context
    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        # Fit the model
        fit_gpytorch_model(mll)
        fit_gpytorch_torch(C_mll, options={"maxiter": 5, "lr": .1, "disp": True})

        # Create a batch
        X_next = generate_batch(
            state=state,
            model=model,
            C_model=C_model,
            X=X_turbo,
            Y=Y_turbo,
            C=C_turbo,
            batch_size=batch_size,
            n_candidates=N_CANDIDATES,
        )

    torch.cuda.empty_cache()
    print("GPU memory:", torch.cuda.memory_allocated(device) / (1 << 30))
    Y_next = torch.tensor([-mopta_evaluate(x)[0] for x in X_next], dtype=dtype, device=device).unsqueeze(-1)
    C_next = torch.stack([mopta_evaluate(x)[1 : n_constraints + 1] for x in X_next], dim=1).to(device).unsqueeze(-1)

    # Update state
    state = update_state(state=state, Y_next=Y_next, C_next=C_next)

    # Append data
    X_turbo = torch.cat((X_turbo, X_next), dim=0)
    Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)
    C_turbo = torch.cat((C_turbo, C_next), dim=1)

    # Print current status
    print(f"{len(X_turbo)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}")
