"""Adversarial perturbation of inputs, used to test explanation robustness
(RQ3 / H3 in RESEARCH_IMPLEMENTATION_PLAN.md).

Three perturbation strategies, chosen for correctness rather than forcing a
single library across incompatible model types:

  - `fgsm_perturb`: single-step gradient attack for the differentiable
    CompactMLP. Cheap, dependency-light, kept as the default/fast option.
  - `deepfool_perturb`: TRUE multiclass DeepFool (Moosavi-Dezfooli et al.,
    2016) for the differentiable CompactMLP -- the actual iterative
    minimum-norm attack used by Munilla & Khammas (2026) against a
    DNN-based IDS, not FGSM's single-step approximation of it. Added so
    this project's robustness results are directly comparable to that
    paper's, which this project's own knowledge-gap synthesis cites as the
    field's reference adversarial-robustness test for IoT-XAI. Since
    DeepFool's natural output is the *minimal* perturbation that flips the
    decision (no epsilon parameter), we clip the accumulated perturbation
    to an L-infinity epsilon ball at every iteration so it still slots into
    this project's existing epsilon-sweep structure for apples-to-apples
    comparison against FGSM at the same epsilon values.
  - `boundary_perturb`: black-box, score-based greedy feature perturbation
    for the non-differentiable LightGBM model, since gradient attacks (both
    FGSM and DeepFool) don't apply to tree ensembles. This is documented
    honestly as a black-box analogue, not mislabeled as DeepFool -- true
    DeepFool has no tree-ensemble equivalent in this project.

All three return perturbed inputs at a given perturbation budget `epsilon`
(L-infinity, in standardized-feature units since data.py z-scores inputs).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fgsm_perturb(mlp_model, X: np.ndarray, y: np.ndarray, epsilon: float) -> np.ndarray:
    mlp_model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    y_t = torch.tensor(y, dtype=torch.long)

    logits = mlp_model(X_t)
    loss = F.cross_entropy(logits, y_t)
    loss.backward()

    with torch.no_grad():
        perturbation = epsilon * X_t.grad.sign()
        X_adv = X_t + perturbation
    return X_adv.detach().numpy()


def deepfool_perturb(
    mlp_model,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    n_classes: int,
    max_iter: int = 20,
    overshoot: float = 0.02,
) -> np.ndarray:
    """True multiclass DeepFool, per-sample, with the accumulated
    perturbation clipped to an L-infinity epsilon ball at every iteration.

    Standard DeepFool: at each step, linearize the decision boundary
    between the current predicted class and every other class, move
    towards the nearest linearized boundary (smallest |f_k - f_true| /
    ||w_k - w_true||), repeat until the label flips or max_iter is hit.
    The epsilon clip means this becomes "the DeepFool direction, capped to
    a fixed L-infinity budget" rather than literature-standard unbounded
    minimal-norm DeepFool -- an explicit, documented adaptation (see module
    docstring) needed to keep it comparable to this project's FGSM sweep at
    matched epsilon values, not a silent deviation from the original attack.
    """
    mlp_model.eval()
    n_samples, n_features = X.shape
    X_adv = X.copy()

    for s in range(n_samples):
        x0 = torch.tensor(X[s : s + 1], dtype=torch.float32)
        x_i = x0.clone()
        true_label = int(y[s])

        for _ in range(max_iter):
            x_i.requires_grad_(True)
            logits = mlp_model(x_i)[0]
            pred_label = int(logits.argmax().item())
            if pred_label != true_label:
                break

            grads = torch.zeros(n_classes, n_features)
            for k in range(n_classes):
                if x_i.grad is not None:
                    x_i.grad.zero_()
                logits = mlp_model(x_i)[0]
                logits[k].backward(retain_graph=True)
                grads[k] = x_i.grad[0].detach()

            f_true = logits[true_label].item()
            w_true = grads[true_label]

            best_ratio = None
            best_w = None
            best_f = None
            for k in range(n_classes):
                if k == true_label:
                    continue
                w_k = grads[k] - w_true
                f_k = (logits[k] - f_true).item()
                w_norm = torch.norm(w_k) + 1e-8
                ratio = abs(f_k) / w_norm.item()
                if best_ratio is None or ratio < best_ratio:
                    best_ratio, best_w, best_f = ratio, w_k, f_k

            with torch.no_grad():
                r_i = (abs(best_f) + 1e-4) / (torch.norm(best_w) ** 2 + 1e-8) * best_w
                step = (1 + overshoot) * r_i
                candidate = x_i.detach() + step
                total_perturbation = torch.clamp(candidate - x0, -epsilon, epsilon)
                x_i = (x0 + total_perturbation).detach()

        X_adv[s] = x_i.detach().numpy()[0]

    return X_adv


def boundary_perturb(
    predict_proba_fn,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    n_features_to_perturb: int = 3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Greedy black-box attack: for each sample, try nudging the
    `n_features_to_perturb` features the current prediction is most
    sensitive to (estimated via finite differences) in the direction that
    reduces true-class probability, within an L-infinity budget epsilon.
    """
    rng = rng or np.random.default_rng(0)
    X_adv = X.copy()
    n_samples, n_features = X.shape
    base_proba = predict_proba_fn(X)

    for s in range(n_samples):
        sensitivities = np.zeros(n_features)
        for f in range(n_features):
            probe = X[s : s + 1].copy()
            probe[0, f] += 1e-2
            p = predict_proba_fn(probe)[0, y[s]]
            sensitivities[f] = base_proba[s, y[s]] - p  # positive = perturbing helps attacker

        top_feats = np.argsort(-sensitivities)[:n_features_to_perturb]
        for f in top_feats:
            direction = 1.0 if sensitivities[f] > 0 else -1.0
            X_adv[s, f] += direction * epsilon

    return X_adv
