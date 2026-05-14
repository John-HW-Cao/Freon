"""
Freon: a family of optimizers based on Schatten (quasi-)norms.

Based on the paper:
  "Muon is Not That Special: Random or Inverted Spectra Work Just as Well"
  arXiv:2605.11181

Three contributions implemented here:

1. **Freon** — Schatten p-quasi-norm optimizer.
   Computes the update direction  U Σ^(p-1) V^T  (where G = U Σ V^T is the SVD
   of the momentum-smoothed gradient), powered by a QDWH-based iterative
   approximation for efficiency.  Special cases:
     p = 1  →  U V^T  (polar factor, equivalent to Muon)
     p = 2  →  G / ‖G‖  (normalised gradient, equivalent to SGD)
   Empirically, the quasi-norm regime (0 < p < 1) works best for GPT-2.

2. **Kaon** — "Absurd" optimizer that replaces singular values with random
   noise while keeping the left/right singular vectors from the SVD of the
   gradient.  Despite lacking any coherent spectral geometry, Kaon matches
   Muon's empirical performance and retains convergence guarantees.

3. **Inverted-spectrum** helper — sorts singular values in the *ascending*
   direction (opposite to the natural descending order), demonstrating that
   the precise ordering of the spectrum is irrelevant.
"""

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Spectral transform functions
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz quintic iteration that approximates the polar factor U V^T.

    This is the p = 1 special case of the Freon transform, and is kept as
    an efficient, SVD-free fast path (identical to the Muon implementation).
    The iteration converges to U S' V^T where S'_{ii} ~ Uniform(0.5, 1.5).

    Reference: KellerJordan/Muon; coefficients maximise slope at zero.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def freon_transform(G: torch.Tensor, p: float) -> torch.Tensor:
    """
    Freon spectral transform: compute  U Σ^(p-1) V^T  from G = U Σ V^T.

    The update direction interpolates between:
      p = 1  →  U V^T          (Muon / polar factor; all σ → 1)
      p = 2  →  U Σ V^T = G   (gradient direction; σ unchanged)
    and extrapolates into the quasi-norm regime:
      0 < p < 1  →  small singular values receive a larger relative boost.

    The paper uses a provably optimal QDWH-based iterative approximation for
    efficiency; this implementation uses exact SVD, which is a correct
    reference implementation (QDWH iterations can be substituted for large
    matrices).

    Args:
        G:  2-D gradient tensor (m × n).
        p:  Schatten exponent (float).  Values in (0, 1) give the quasi-norm
            regime; p = 1 is Muon; p = 2 is SGD.

    Returns:
        2-D tensor of the same shape as G with spectral norm ≈ 1, scaled by
        sqrt(max(m, n)) for consistency with Muon's normalisation.
    """
    assert G.ndim == 2, "freon_transform expects a 2-D matrix"
    m, n = G.shape

    # Normalise by Frobenius norm for numerical stability
    G_f = G.float()
    G_f = G_f / (G_f.norm() + 1e-7)

    U, S, Vh = torch.linalg.svd(G_f, full_matrices=False)

    # Apply Schatten p transform: σ_i → σ_i^(p-1)
    S_p = S.clamp(min=1e-7).pow(p - 1)

    # Normalise so that the spectral norm equals 1
    S_p = S_p / (S_p.max() + 1e-7)

    out = (U * S_p.unsqueeze(0)) @ Vh
    # Aspect-ratio scaling — same convention as Muon's muon_update
    out = out * max(1, m / n) ** 0.5
    return out.to(G.dtype)


def kaon_transform(G: torch.Tensor) -> torch.Tensor:
    """
    Kaon spectral transform: replace singular values of G with random noise.

    Keeps U and V from the SVD of G but draws the diagonal of Σ i.i.d. from
    Uniform(0, 1), demonstrating that the specific spectrum is irrelevant for
    optimisation performance.

    Args:
        G:  2-D gradient tensor (m × n).

    Returns:
        2-D tensor of the same shape as G with spectral norm ≈ 1, scaled by
        sqrt(max(m, n)).
    """
    assert G.ndim == 2, "kaon_transform expects a 2-D matrix"
    m, n = G.shape

    G_f = G.float()
    G_f = G_f / (G_f.norm() + 1e-7)

    U, _S, Vh = torch.linalg.svd(G_f, full_matrices=False)

    # Replace singular values with i.i.d. Uniform(0, 1) noise
    S_rand = torch.rand(_S.shape, device=G.device, dtype=torch.float32)
    S_rand = S_rand / (S_rand.max() + 1e-7)

    out = (U * S_rand.unsqueeze(0)) @ Vh
    out = out * max(1, m / n) ** 0.5
    return out.to(G.dtype)


def inverted_transform(G: torch.Tensor) -> torch.Tensor:
    """
    Inverted-spectrum transform: reverse the order of singular values.

    SVD returns singular values in descending order σ_1 ≥ σ_2 ≥ …; this
    function pairs the largest left/right singular vector *pair* with the
    *smallest* singular value, and so on.  Performance matches Muon despite
    the structural inversion, confirming that spectral ordering is irrelevant.

    Args:
        G:  2-D gradient tensor (m × n).

    Returns:
        2-D tensor of the same shape as G, scaled by sqrt(max(m, n)).
    """
    assert G.ndim == 2, "inverted_transform expects a 2-D matrix"
    m, n = G.shape

    G_f = G.float()
    G_f = G_f / (G_f.norm() + 1e-7)

    U, S, Vh = torch.linalg.svd(G_f, full_matrices=False)

    # Flip singular values: descending → ascending
    S_inv = S.flip(0)
    S_inv = S_inv / (S_inv.max() + 1e-7)

    out = (U * S_inv.unsqueeze(0)) @ Vh
    out = out * max(1, m / n) ** 0.5
    return out.to(G.dtype)


# ---------------------------------------------------------------------------
# Per-parameter update helpers (momentum + spectral transform)
# ---------------------------------------------------------------------------

def freon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    p: float = 0.5,
    nesterov: bool = True,
) -> torch.Tensor:
    """
    Compute one Freon update step (momentum + Schatten p transform).

    Args:
        grad:     Raw gradient (any shape with ndim ≥ 2).
        momentum: Running momentum buffer (same shape as grad, mutated).
        beta:     Momentum coefficient.
        p:        Schatten exponent for the spectral transform.
        nesterov: Use Nesterov-style momentum (default True).

    Returns:
        Update tensor (2-D; caller is responsible for reshaping to grad.shape).
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    # Collapse higher-dimensional tensors (e.g. conv filters) to 2-D
    if update.ndim > 2:
        update = update.view(update.shape[0], -1)

    if p == 1.0:
        # Fast path: use Newton-Schulz (no SVD required)
        update = zeropower_via_newtonschulz5(update)
        update = update * max(1, update.size(-2) / update.size(-1)) ** 0.5
    else:
        # freon_transform already applies the aspect-ratio scaling
        update = freon_transform(update, p)

    return update


def kaon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    nesterov: bool = True,
) -> torch.Tensor:
    """
    Compute one Kaon update step (momentum + random-spectrum transform).

    Args:
        grad:     Raw gradient (any shape with ndim ≥ 2).
        momentum: Running momentum buffer (same shape as grad, mutated).
        beta:     Momentum coefficient.
        nesterov: Use Nesterov-style momentum (default True).

    Returns:
        Update tensor (2-D; caller is responsible for reshaping to grad.shape).
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    if update.ndim > 2:
        update = update.view(update.shape[0], -1)

    update = kaon_transform(update)
    return update


# ---------------------------------------------------------------------------
# Distributed Freon optimizer
# ---------------------------------------------------------------------------

class Freon(torch.optim.Optimizer):
    """
    Freon — Schatten quasi-norm optimizer (distributed variant).

    Freon generalises Muon by parameterising the spectral update with a
    Schatten exponent p:

      update = U Σ^(p-1) V^T        (G = U Σ V^T)

    Special values:
      p = 1  →  U V^T  (Muon; use SingleDeviceMuon / Muon for a faster path)
      p = 2  →  G      (gradient; equivalent to normalised SGD)
      0 < p < 1  →  quasi-norm regime, empirically best for GPT-2

    This class mirrors the structure of KellerJordan/Muon (distributed with
    dist.all_gather for parameter synchronisation).  For single-GPU training
    use SingleDeviceFreon.

    Arguments:
        params:        List of nn.Parameter objects (hidden weight matrices).
        lr:            Learning rate (spectral-norm units per step).
        weight_decay:  AdamW-style weight decay coefficient.
        momentum:      SGD momentum coefficient (default 0.95).
        p:             Schatten exponent (default 0.5, quasi-norm regime).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        p: float = 0.5,
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, p=p)
        assert (
            isinstance(params, list)
            and len(params) >= 1
            and isinstance(params[0], torch.nn.Parameter)
        )
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            world_size = dist.get_world_size()
            params_pad = params + [torch.empty_like(params[-1])] * (
                world_size - len(params) % world_size
            )
            for base_i in range(len(params))[::world_size]:
                if base_i + dist.get_rank() < len(params):
                    param = params[base_i + dist.get_rank()]
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    state = self.state[param]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(param)
                    upd = freon_update(
                        param.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        p=group["p"],
                    )
                    param.mul_(1 - group["lr"] * group["weight_decay"])
                    param.add_(upd.reshape(param.shape), alpha=-group["lr"])
                dist.all_gather(
                    params_pad[base_i : base_i + world_size],
                    params_pad[base_i + dist.get_rank()],
                )

        return loss


# ---------------------------------------------------------------------------
# Single-device Freon optimizer
# ---------------------------------------------------------------------------

class SingleDeviceFreon(torch.optim.Optimizer):
    """
    Freon — Schatten quasi-norm optimizer (single-device variant).

    See :class:`Freon` for full documentation.  This version does not require
    an initialised process group and is suitable for single-GPU / CPU training.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        p: float = 0.5,
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, p=p)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                state = self.state[param]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(param)
                upd = freon_update(
                    param.grad,
                    state["momentum_buffer"],
                    beta=group["momentum"],
                    p=group["p"],
                )
                param.mul_(1 - group["lr"] * group["weight_decay"])
                param.add_(upd.reshape(param.shape), alpha=-group["lr"])

        return loss


# ---------------------------------------------------------------------------
# Distributed Kaon optimizer
# ---------------------------------------------------------------------------

class Kaon(torch.optim.Optimizer):
    """
    Kaon — random-spectrum optimizer (distributed variant).

    Kaon replaces the singular values of the momentum-smoothed gradient with
    i.i.d. Uniform(0, 1) noise while keeping the left/right singular vectors.
    Despite the lack of coherent spectral geometry, Kaon matches Muon's
    empirical training performance, demonstrating that the precise singular
    value distribution is not the key driver of optimisation success.

    This class uses the same distributed synchronisation pattern as Muon /
    Freon.  For single-GPU training use SingleDeviceKaon.

    Arguments:
        params:        List of nn.Parameter objects (hidden weight matrices).
        lr:            Learning rate.
        weight_decay:  AdamW-style weight decay coefficient.
        momentum:      SGD momentum coefficient (default 0.95).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert (
            isinstance(params, list)
            and len(params) >= 1
            and isinstance(params[0], torch.nn.Parameter)
        )
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            world_size = dist.get_world_size()
            params_pad = params + [torch.empty_like(params[-1])] * (
                world_size - len(params) % world_size
            )
            for base_i in range(len(params))[::world_size]:
                if base_i + dist.get_rank() < len(params):
                    param = params[base_i + dist.get_rank()]
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    state = self.state[param]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(param)
                    upd = kaon_update(
                        param.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                    )
                    param.mul_(1 - group["lr"] * group["weight_decay"])
                    param.add_(upd.reshape(param.shape), alpha=-group["lr"])
                dist.all_gather(
                    params_pad[base_i : base_i + world_size],
                    params_pad[base_i + dist.get_rank()],
                )

        return loss


# ---------------------------------------------------------------------------
# Single-device Kaon optimizer
# ---------------------------------------------------------------------------

class SingleDeviceKaon(torch.optim.Optimizer):
    """
    Kaon — random-spectrum optimizer (single-device variant).

    See :class:`Kaon` for full documentation.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                state = self.state[param]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(param)
                upd = kaon_update(
                    param.grad,
                    state["momentum_buffer"],
                    beta=group["momentum"],
                )
                param.mul_(1 - group["lr"] * group["weight_decay"])
                param.add_(upd.reshape(param.shape), alpha=-group["lr"])

        return loss
