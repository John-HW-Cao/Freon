# Freon

**Freon** is a family of optimizers based on Schatten (quasi-)norms, powered by a
novel, provably optimal QDWH-based iterative approximation.

Introduced in:
> *Muon is Not That Special: Random or Inverted Spectra Work Just as Well*  
> arXiv:2605.11181

---

## Optimizers

### Freon

Freon generalises [Muon](https://github.com/KellerJordan/Muon) by parameterising
the spectral update with a Schatten exponent `p`:

```
update = U Σ^(p-1) V^T   (where G = U Σ V^T is the gradient SVD)
```

| `p` | Behaviour |
|-----|-----------|
| `p = 1` | Muon (polar factor `U V^T`, all singular values → 1) |
| `p = 2` | Normalised SGD (gradient direction) |
| `0 < p < 1` | Quasi-norm regime — empirically best for GPT-2 |

### Kaon

Kaon is the "absurd" optimizer from the paper: it keeps `U` and `V` from the SVD
of the gradient but replaces the singular values with i.i.d. Uniform(0, 1) noise.
Despite having no coherent spectral geometry, Kaon matches Muon's training
performance, demonstrating that precise singular-value structure is not the key
driver of optimisation success.

---

## Installation

```bash
pip install -e .
```

## Usage

Both `SingleDeviceFreon` and `SingleDeviceKaon` work on a single GPU or CPU.
The distributed `Freon` and `Kaon` classes mirror the `Muon` distributed API
(using `dist.all_gather` for parameter synchronisation).

```python
from freon import SingleDeviceFreon, SingleDeviceKaon

# Freon with default quasi-norm exponent p=0.5 (best for GPT-2)
hidden_params = [p for n, p in model.named_parameters() if p.ndim >= 2]
optimizer = SingleDeviceFreon(hidden_params, lr=0.02, momentum=0.95, p=0.5)

# Kaon (random-spectrum variant)
optimizer = SingleDeviceKaon(hidden_params, lr=0.02, momentum=0.95)
```

### Distributed training

```python
from freon import Freon, Kaon

hidden_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2]
optimizer = Freon(hidden_params, lr=0.02, momentum=0.95, p=0.5)
```

---

## API

### `freon_transform(G, p)`

Core Freon spectral transform.  Computes `U Σ^(p-1) V^T` from a 2-D gradient
matrix `G = U Σ V^T` using exact SVD.  (A provably optimal QDWH-based iterative
approximation can be substituted for efficiency on large matrices.)

### `kaon_transform(G)`

Core Kaon transform.  Returns `U Ξ V^T` where `Ξ` contains i.i.d. Uniform(0,1)
random values and `U`, `V` come from the SVD of `G`.

### `inverted_transform(G)`

Helper that flips the singular-value ordering (largest ↔ smallest), demonstrating
that even the order of the spectrum is irrelevant for optimisation performance.

---

## Paper summary

The paper makes three contributions:

1. **Freon** interpolates between SGD and Muon via the Schatten p-quasi-norm and
   outperforms Muon in the quasi-norm regime (p < 1) for GPT-2, a regime that
   cannot be represented by any unitarily invariant LMO.

2. **Kaon** shows that replacing singular values with random noise still matches
   Muon, proving that strict adherence to a precise spectral geometry is
   practically irrelevant.

3. The paper shows that optimisation performance is controlled by two *local*
   quantities — **alignment** and **descent potential** — rather than global
   geometric structure, explaining why Muon succeeds by guaranteeing step-size
   optimality rather than by tracking an ideal global geometry.

