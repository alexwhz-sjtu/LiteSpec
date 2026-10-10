from __future__ import annotations

import math


def optimizer_step_count(phases, steps_per_epoch, accumulation_steps):
    """Each phase flushes its own tail; epochs within a phase do not."""
    return sum(
        math.ceil(
            (
                phase.epochs * steps_per_epoch
                if phase.schedule_full_epochs
                else phase.batch_count(steps_per_epoch)
            )
            / accumulation_steps
        )
        for phase in phases
    )


import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR as _CosineAnnealingLR
from torch.optim.lr_scheduler import LRScheduler as _LRScheduler

from litespec.distributed import print_on_rank0


class _enable_get_lr_call:
    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_lr_called_within_step = False


class TwoStageScheduler(_LRScheduler):
    def __init__(self, optimizer, after_scheduler: _LRScheduler, last_epoch=-1):
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def state_dict(self):
        state_dict = {
            key: value for key, value in self.__dict__.items() if key not in "optimizer"
        }
        if isinstance(state_dict["after_scheduler"], _LRScheduler):
            state_dict["after_scheduler_type"] = type(
                state_dict["after_scheduler"]
            ).__name__
            state_dict["after_scheduler_dict"] = state_dict[
                "after_scheduler"
            ].state_dict()
            del state_dict["after_scheduler"]
        else:
            raise NotImplementedError()
        return state_dict

    def load_state_dict(self, state_dict):
        expected = type(self.after_scheduler).__name__
        if state_dict["after_scheduler_type"] != expected:
            raise ValueError(f"Scheduler type must be {expected}")
        self.after_scheduler.load_state_dict(state_dict["after_scheduler_dict"])
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key not in ("after_scheduler_type", "after_scheduler_dict")
        }
        super().load_state_dict(state_dict)


class DelayerScheduler(TwoStageScheduler):
    """Starts with a flat lr schedule until it reaches N epochs then applies
    the specific scheduler (For example: ReduceLROnPlateau)

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        delay_epochs (int): Number of epochs to keep the initial lr until starting applying the scheduler.
        after_scheduler (:class:`torch.optim.lr_scheduler`): After target_epoch, use this scheduler.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, delay_epochs, after_scheduler, last_epoch=-1):
        if delay_epochs < 0:
            raise ValueError(f"delay_epochs must >= 0, got {delay_epochs}")
        self.delay_epochs = delay_epochs
        super().__init__(optimizer, after_scheduler, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.delay_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            with _enable_get_lr_call(self.after_scheduler):
                return self.after_scheduler.get_lr()

        return self.base_lrs

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                self.after_scheduler.step(epoch - self.delay_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(DelayerScheduler, self).step(epoch)


class WarmupScheduler(TwoStageScheduler):
    """Starts with a linear warmup lr schedule until it reaches N epochs then applies
    the specific scheduler (For example: ReduceLROnPlateau).

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_epochs (int): Number of epochs to linearly warmup lr until starting applying the scheduler.
        after_scheduler (:class:`torch.optim.lr_scheduler`): After target_epoch, use this scheduler.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):
        self.warmup_epochs = int(warmup_epochs)
        super().__init__(optimizer, after_scheduler, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()

        return [(self.last_epoch + 1) / self.warmup_epochs * lr for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)


class WarmupDelayerScheduler(TwoStageScheduler):
    """Starts with a linear warmup lr schedule until it reaches N epochs and a flat lr schedule
    until it reaches M epochs then applies the specific scheduler (For example: ReduceLROnPlateau).

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_epochs (int): Number of epochs to linearly warmup lr until starting applying the scheduler.
        delay_epochs (int): Number of epochs to keep the initial lr until starting applying the scheduler.
        after_scheduler (:class:`torch.optim.lr_scheduler`): After target_epoch, use this scheduler.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(
        self, optimizer, warmup_epochs, delay_epochs, after_scheduler, last_epoch=-1
    ):
        if delay_epochs < 0:
            raise ValueError(f"delay_epochs must >= 0, got {delay_epochs}")
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must >= 0, got {warmup_epochs}")
        self.warmup_epochs = warmup_epochs
        self.delay_epochs = delay_epochs
        super().__init__(optimizer, after_scheduler, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs + self.delay_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                # reset lr to base_lr
                for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                    group["lr"] = base_lr
                self.finished = True
            with _enable_get_lr_call(self.after_scheduler):
                return self.after_scheduler.get_lr()
        elif self.last_epoch >= self.warmup_epochs:
            return self.base_lrs

        return [(self.last_epoch + 1) / self.warmup_epochs * lr for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)


class CosineAnnealingLR(_CosineAnnealingLR):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        eta_min (int, optional): Minimum learning rate, defaults to 0.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        eta_min: int = 0,
        last_epoch: int = -1,
        **kwargs,
    ):
        super().__init__(optimizer, total_steps, eta_min=eta_min, last_epoch=last_epoch)


class CosineAnnealingWarmupLR(WarmupScheduler):
    """Cosine annealing learning rate scheduler with learning rate warmup. A linear warmup schedule will be applied.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0.
        eta_min (int, optional): Minimum learning rate, defaults to 0.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        base_scheduler = _CosineAnnealingLR(
            optimizer,
            max(total_steps - warmup_steps, 1),
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)


class BF16Optimizer:
    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
        parameters=None,
        process_group=None,
    ):
        # TODO: For now, we only support cosine annealing warmup lr scheduler and AdamW optimizer
        # TODO: We should make these parameters configurable
        #   These magic numbers: weight_decay=0.0, max_grad_norm=0.5, total_steps=800k, warmup_steps=12k are copied from
        self.model = model
        self.model_params = (
            [p for p in model.parameters() if p.requires_grad]
            if parameters is None
            else list(parameters)
        )
        if not self.model_params:
            raise ValueError("BF16Optimizer requires at least one model parameter.")
        parameter_names = {
            id(parameter): name for name, parameter in model.named_parameters()
        }
        self.parameter_names = [
            parameter_names[id(parameter)] for parameter in self.model_params
        ]
        self.optimizer_config = dict(
            lr=float(lr),
            weight_decay=float(weight_decay),
            max_grad_norm=float(max_grad_norm),
            total_steps=int(total_steps),
            warmup_ratio=float(warmup_ratio),
        )
        self.max_grad_norm = float(max_grad_norm)
        self.process_group = process_group
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {max_grad_norm}.")
        self.fp32_params = [
            p.detach().clone().to(torch.float32) for p in self.model_params
        ]
        for mp in self.fp32_params:
            mp.requires_grad = True
        self.optimizer = torch.optim.AdamW(
            self.fp32_params, lr=lr, weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingWarmupLR(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
        )

    def zero_grad(self) -> None:
        """Clear both FSDP model gradients and FP32 optimizer gradients."""
        self.optimizer.zero_grad(set_to_none=True)
        for param in self.model_params:
            param.grad = None
        for param in self.fp32_params:
            param.grad = None

    def scale_model_gradients(self, factor: float) -> None:
        """Scale accumulated FSDP gradients before they are copied to FP32."""
        if factor <= 0:
            raise ValueError(f"gradient scale must be positive, got {factor}.")
        with torch.no_grad():
            for param in self.model_params:
                if param.grad is not None:
                    param.grad.mul_(factor)

    def _global_grad_norm(self) -> torch.Tensor:
        """Return the global L2 norm using one scalar all-reduce.

        Gradients remain sharded and local. Only their local squared L2 norm is
        summed across ranks.
        """
        if not self.fp32_params:
            return torch.tensor(0.0)

        device = self.fp32_params[0].device
        gradients = [param.grad for param in self.fp32_params if param.grad is not None]
        if gradients:
            local_norms = torch._foreach_norm(gradients, 2)
            local_squared_norm = (
                torch.stack([norm.double() for norm in local_norms]).square().sum()
            )
        else:
            local_squared_norm = torch.zeros((), device=device, dtype=torch.float64)

        if dist.is_available() and dist.is_initialized():
            # Communication payload: exactly one 0-D scalar, never gradients.
            dist.all_reduce(
                local_squared_norm,
                op=dist.ReduceOp.SUM,
                group=self.process_group,
            )
        return local_squared_norm.sqrt()

    def step(self) -> float:
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                mp.grad = (
                    p.grad.detach().to(torch.float32) if p.grad is not None else None
                )

        grad_norm = self._global_grad_norm()
        grad_norm_value = float(grad_norm.item())
        clip_coefficient = min(
            self.max_grad_norm / (grad_norm_value + 1e-6),
            1.0,
        )
        if clip_coefficient < 1.0:
            with torch.no_grad():
                for param in self.fp32_params:
                    if param.grad is not None:
                        param.grad.mul_(clip_coefficient)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.data.copy_(mp.data.to(p.dtype))
                p.grad = None
        return grad_norm_value

    def _param_state_compatible(self, param, param_state) -> bool:
        for key, value in param_state.items():
            if key == "step" or not isinstance(value, torch.Tensor):
                continue
            if value.shape != param.shape:
                return False
        return True

    def _optimizer_state_compatible(self, optimizer_state_dict) -> bool:
        """Return True when every saved Adam moment matches current fp32 param shapes."""
        ckpt_params = optimizer_state_dict["param_groups"][0]["params"]
        if len(ckpt_params) != len(self.fp32_params):
            print_on_rank0(
                "Optimizer checkpoint has "
                f"{len(ckpt_params)} params, expected {len(self.fp32_params)}."
            )
            return False

        for idx, (param, param_id) in enumerate(
            zip(self.fp32_params, ckpt_params, strict=True)
        ):
            param_state = optimizer_state_dict["state"].get(param_id)
            if param_state is None:
                continue
            if not self._param_state_compatible(param, param_state):
                print_on_rank0(
                    "Optimizer state shape mismatch at param index "
                    f"{idx}: checkpoint moments do not match model "
                    f"{tuple(param.shape)} ({param.numel()} elements)."
                )
                return False
        return True

    def load_state_dict(self, state_dict) -> bool:
        if state_dict["optimizer_config"] != self.optimizer_config:
            raise ValueError(
                "Optimizer/scheduler configuration does not match the checkpoint"
            )
        ckpt_opt = state_dict["optimizer_state_dict"]
        if state_dict["optimizer_parameter_names"] != self.parameter_names:
            raise ValueError(
                "Optimizer parameter names/order do not match the checkpoint"
            )
        if not self._optimizer_state_compatible(ckpt_opt):
            raise ValueError("Optimizer parameter shapes do not match the checkpoint")
        masters = state_dict["fp32_master_params"]
        if len(masters) != len(self.fp32_params) or any(
            source.shape != target.shape or source.dtype != torch.float32
            for source, target in zip(masters, self.fp32_params)
        ):
            raise ValueError(
                "FP32 master parameter shapes/dtypes do not match the checkpoint"
            )
        with torch.no_grad():
            for model_param, master, saved in zip(
                self.model_params, self.fp32_params, masters, strict=True
            ):
                master.copy_(saved)
                model_param.copy_(master.to(model_param.dtype))
        self.optimizer.load_state_dict(ckpt_opt)
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        print_on_rank0("Successfully loaded optimizer and scheduler state.")
        return True

    def state_dict(self):
        return {
            "optimizer_parameter_names": self.parameter_names,
            "optimizer_config": self.optimizer_config,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "fp32_master_params": [p.detach().cpu().clone() for p in self.fp32_params],
        }

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]


def normalize_accumulated_gradients(
    optimizer,
    local_denominator: torch.Tensor,
    accumulation_steps: int,
    group=None,
) -> float:
    """Normalize rank-averaged gradients by the global optimizer-window weight.

    Every microbatch backpropagates its additive loss numerator divided by the
    configured accumulation length. FSDP averages those gradients across ranks.
    Multiplying by ``world_size * accumulation_steps / global_denominator``
    therefore produces the gradient of the summed numerator divided by the sum
    of effective weights across every microbatch and DP rank in the window.
    """
    if int(accumulation_steps) <= 0:
        raise ValueError("accumulation_steps must be positive")
    denominator = local_denominator.detach().float().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size(group=group)
        if world_size > 1:
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=group)
    denominator_value = float(denominator.item())
    if not math.isfinite(denominator_value) or denominator_value <= 0:
        raise ValueError("global loss denominator must be finite and positive")
    scale = world_size * int(accumulation_steps) / denominator_value
    optimizer.scale_model_gradients(scale)
    return denominator_value
