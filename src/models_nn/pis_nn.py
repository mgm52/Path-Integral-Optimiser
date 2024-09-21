import copy

import numpy as np
import torch as th
from torch import nn

from src.models_nn.time_conder import TimeConder
from src.utils.loss_helper import nll_unit_gaussian

import pdb

def get_reg_fns(fns=None):
    from jammy.utils import imp

    reg_fns = []
    if fns is None:
        return reg_fns

    for _fn in fns:
        reg_fns.append(imp.load_class(_fn))

    return reg_fns


# pylint: disable=function-redefined


class PISNN(nn.Module):  # pylint: disable=abstract-method, too-many-instance-attributes
    def __init__(
        self,
        f_func,
        g_func,
        reg_fns, # e.g. quad_reg
        grad_fn=None,
        f_format="f",
        g_coef=np.sqrt(0.2), # TODO: figure out where this comes from...? and whether it should be 1/K instead, where K is num of timesteps. or hopefully sdeint is doing that for me.
        data_shape=2,
        t_end=1.0,
        sde_type="stratonovich",
        noise_type="diagonal",
        nn_clip=1e2,
        lgv_clip=1e2,
        sigma_rescaling="static",
        get_sigma=None
    ):  # pylint: disable=too-many-arguments
        super().__init__()
        self.f_func = f_func
        self.g_func = g_func
        self.reg_fns = get_reg_fns(reg_fns)
        self.nreg = len(reg_fns)
        self.sde_type = sde_type
        self.noise_type = noise_type
        self.nn_clip = nn_clip * 1.0
        self.lgv_clip = lgv_clip * 1.0
        self.data_ndim = np.prod(data_shape)
        self.data_shape = (
            tuple(
                [
                    data_shape,
                ]
            )
            if isinstance(data_shape, int)
            else data_shape
        )
        # BaseModel sets grad_fn to trainer.datamodule.dataset.score (i.e. grad of NLL in target distribution).
        self.grad_fn = grad_fn
        self.g_coef = g_coef

        self.t_end = t_end
        self.select_f(f_format)

        self.sigma_rescaling = sigma_rescaling
        self.get_sigma = get_sigma

        if self.sigma_rescaling in ["static","dynamic"]:
            self.sigma_factor = self.get_sigma()
            self.sqrt_sigma_factor = np.sqrt(self.get_sigma())
        elif self.sigma_rescaling == "none":
            self.sigma_factor = 1.0
            self.sqrt_sigma_factor = 1.0
        else:
            raise RuntimeError()

    def select_f(self, f_format=None):
        _fn = self.f_func
        if f_format == "f":

            def _fn(t, x):
                return th.clip(self.f_func(t, x), -self.nn_clip, self.nn_clip)

        # base.yaml default is t_tnet_grad
        elif f_format == "t_tnet_grad":
            self.lgv_coef = TimeConder(64, 1, 3)

            def _fn(t, x):
                # grad is the grad of the NLL of target distribution wrt x
                # self.lgv_coef constant
                grad = th.clip(self.grad_fn(x), -self.lgv_clip, self.lgv_clip)
                f = th.clip(self.f_func(t, x), -self.nn_clip, self.nn_clip)

                if (not th.isfinite(grad).all()) or abs(grad).max() == self.lgv_clip:
                    print(f"    (WARNING) For t={t}: Taking grad.min={round(float(grad.min()), -3)}, grad.max={round(float(grad.max()), -3)}")
                    # UNCOMMENT TO PLOT GRADIENTS
                    # Plot chart of all x values, with the y axis being their corresponding grad
                    # Note that x and grad are tensors of shape (6000, 1), so we need to convert them to numpy arrays
                    
                    #import matplotlib.pyplot as plt
                    #x_np = x.cpu().detach().numpy()
                    #grad_np = grad.cpu().detach().numpy()
                    # Replace nan values with 20000
                    #x_np = np.nan_to_num(x_np, nan=20000)
                    #grad_np = np.nan_to_num(grad_np, nan=20000)

                    #plt.scatter(x_np, grad_np)
                    #plt.savefig(f"grad_{t}.pdf")
                    #pdb.set_trace()

                return f - self.lgv_coef(t) * grad

        elif f_format == "nn_grad":

            def _fn(t, x):
                x_dot = th.clip(self.grad_fn(x), -self.lgv_clip, self.lgv_clip)
                f_x = th.clip(self.f_func(t, x), -self.nn_clip, self.nn_clip)
                return f_x * x_dot

        elif f_format == "comp_grad":
            self.grad_net = copy.deepcopy(self.f_func)

            def _fn(t, x):
                x_dot = th.clip(self.grad_fn(x), -self.lgv_clip, self.lgv_clip)
                f_x = th.clip(self.f_func(t, x), -self.nn_clip, self.nn_clip)
                f_x_dot = th.clip(self.grad_net(t, x_dot), -self.nn_clip, self.nn_clip)
                return f_x + f_x_dot

        else:
            raise RuntimeError

        self.param_fn = _fn

    # Drift term u_t(x). Used in sdeint.
    def f(self, t, state):
        # t: scaler
        # state: Tensor of shape (n_trajectories, data_ndim + n_reg)
        class SharedContext:  # pylint: disable=too-few-public-methods
            pass

        x = th.nan_to_num(state[:, : -self.nreg])
        x = x.view(-1, *self.data_shape)

        if self.sigma_rescaling == "dynamic":
            self.sigma_factor = self.get_sigma()

        control = self.sigma_factor * self.param_fn(t, x).flatten(start_dim=1)
        dreg = tuple(reg_fn(x, control, SharedContext) for reg_fn in self.reg_fns)
        #print(f"    (INFO) For t={t}: Taking control.min={round(float(control.min()), -3)}, control.max={round(float(control.max()), -3)}")
        # Note: why do we multiply by g_coef here?
        return th.cat((control * self.g_coef,) + dreg, dim=1)

    # Noise term w_t. Used in sdeint.
    # Output should be same shape, unless brownian motion multidimensional
    # https://github.com/google-research/torchsde/blob/53038a3efcd77f6c9f3cfd0310700a59be5d5d2d/torchsde/_core/sdeint.py#L46
    def g(self, t, state):
        # t: scaler
        # state: Tensor of shape (n_trajectories, data_ndim + n_reg)
        if self.sigma_rescaling == "dynamic":
            self.sqrt_sigma_factor = np.sqrt(self.get_sigma())

        origin_g = self.sqrt_sigma_factor * self.g_func(t, state[:, : -self.nreg]) * self.g_coef
        return th.cat(
            (origin_g, th.zeros((state.shape[0], self.nreg)).to(origin_g)), dim=1
        )

    def zero(self, batch_size, device="cpu"):
        return th.zeros(batch_size, self.data_ndim + self.nreg, device=device)

    # NLL of uncontrolled process mu_0(x)
    def nll_prior(self, state):
        state = state[:, : self.data_ndim]
        return nll_unit_gaussian(state, np.sqrt(self.t_end) * self.g_coef)

    # Returns drift term u_t(x) and noise term w_t
    def f_and_g_prod(self, t, y, v):
        v[:, -self.nreg :] = v[:, -self.nreg :] * 0
        return self.f(t, y), v * self.g_coef

    # state[:,:,-n_reg]: Previous output state, without reg terms (x_t)
    # f_value:           Drift term, scaled down by g_coef (u_t(x)) [but with an added regularization term]
    # g_prod_noise:      Noise term, scaled down by g_coef (w_t)
    # uw_term:           (u_t(x) * w_t) term - used for computing loss & importance weights in sampling
    # NOTE that in training (not visualization), this is handled by torchsde.sdeint instead
    def step_with_uw(self, t, state, dt):
        #print(f"Prev state min is {state.min()}, max is {state.max()} at t={t}")

        # Noise term w_t
        noise = th.randn_like(state) * np.sqrt(dt)

        f_value, g_prod_noise = self.f_and_g_prod(t, state, noise)
        new_state = state + f_value * dt + g_prod_noise
        uw_term = (f_value[:, :-1] * noise[:, :-1]).sum(dim=1) / self.g_coef

        return new_state, uw_term
