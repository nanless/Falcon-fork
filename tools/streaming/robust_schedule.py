"""Warmup and validation scheduling for large-batch speech training."""
from __future__ import annotations
import math
import torch
import torch.distributed as dist


class RobustWarmupPlateau:
    """Track actual best from the first validation and reduce LR during warmup.

    Values are minimized.  `best` is the exact checkpoint-selection minimum;
    `plateau_best` uses min_delta to define a significant improvement. A
    severe absolute degradation triggers an immediate scale reduction. This
    state machine does not itself skip a batch or restore a checkpoint.
    """
    def __init__(self, target_lr, warmup_steps, *, factor=0.5, patience=5,
                 min_delta=0.005, severe_delta=10.0, min_scale=1e-4):
        if not math.isfinite(target_lr) or target_lr <= 0 or warmup_steps < 0:
            raise ValueError("invalid target learning rate or warmup")
        if not 0 < factor < 1 or patience < 0 or min_delta < 0 or severe_delta <= 0 or not 0 < min_scale <= 1:
            raise ValueError("invalid plateau configuration")
        self.target_lr = float(target_lr)
        self.warmup_steps = int(warmup_steps)
        self.factor = float(factor)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.severe_delta = float(severe_delta)
        self.min_scale = float(min_scale)
        self.scale = 1.0
        self.best = math.inf
        self.plateau_best = math.inf
        self.bad_epochs = 0
        self.validations = 0

    def learning_rate(self, completed_updates):
        if completed_updates < 0:
            raise ValueError("completed updates must be nonnegative")
        fraction = min((completed_updates + 1) / self.warmup_steps, 1.0) if self.warmup_steps else 1.0
        return self.target_lr * fraction * self.scale

    def observe(self, validation):
        validation = float(validation)
        if not math.isfinite(validation):
            raise ValueError("validation loss must be finite")
        improved = validation < self.best
        self.best = min(self.best, validation)
        severe = validation > self.plateau_best + self.severe_delta
        if validation < self.plateau_best - self.min_delta:
            self.plateau_best = validation
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        reduced = severe or self.bad_epochs > self.patience
        if reduced:
            old_scale = self.scale
            self.scale = max(self.min_scale, self.scale * self.factor)
            reduced = self.scale < old_scale
            self.bad_epochs = 0
        self.validations += 1
        return {"improved": improved, "severe": severe, "reduced": reduced,
                "best": self.best, "scale": self.scale}

    def reduce_after_bad_step(self):
        """Reduce future learning rates after a globally rejected update."""
        old_scale = self.scale
        self.scale = max(self.min_scale, self.scale * self.factor)
        return self.scale < old_scale

    def state_dict(self):
        return {key: getattr(self, key) for key in (
            "target_lr", "warmup_steps", "factor", "patience", "min_delta",
            "severe_delta", "min_scale", "scale", "best", "plateau_best",
            "bad_epochs", "validations")}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if set(state) != set(expected):
            raise ValueError("schedule state fields do not match")
        config_keys = ("target_lr", "warmup_steps", "factor", "patience", "min_delta", "severe_delta", "min_scale")
        if any(state[k] != expected[k] for k in config_keys):
            raise ValueError("schedule configuration differs")
        if not self.min_scale <= state["scale"] <= 1 or state["bad_epochs"] < 0 or state["validations"] < 0:
            raise ValueError("invalid schedule state")
        if any(math.isnan(state[k]) for k in ("best", "plateau_best")):
            raise ValueError("invalid saved validation state")
        for k in ("scale", "best", "plateau_best", "bad_epochs", "validations"):
            setattr(self, k, state[k])


def global_gradient_norm(norm):
    """Return maximum unclipped norm over ranks; NaN is an infinite failure."""
    value = torch.nan_to_num(norm.detach().clone(), nan=float("inf"), posinf=float("inf"))
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value
