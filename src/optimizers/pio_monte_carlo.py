from collections import deque
import math
import torch
from torch.optim import _functional as F
from torch.optim.optimizer import Optimizer, required
import torch as th

# Under the hood, Lightning does the following:

#     for epoch in epochs:
#         for batch in data:

#             def closure():
#                 loss = model.training_step(batch, batch_idx)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 return loss

#             optimizer.step(closure)

#         lr_scheduler.step()

# As can be seen in the code snippet above, Lightning defines a closure with ``training_step()``, ``optimizer.zero_grad()``
# and ``loss.backward()`` for the optimization. This mechanism is in place to support optimizers which operate on the
# output of the closure (e.g. the loss) or need to call the closure several times (e.g. :class:`~torch.optim.LBFGS`).

class PIOMonteCarlo(Optimizer):
    r"""Implements PIO Monte Carlo

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.PIOMonteCarlo(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, sigma=required, mc_max_steps_total=required, mc_ts_per_mc_step=required, sigma_rescaling=True, max_grad_norm=None):

        if sigma is not required and sigma < 0.0:
            raise ValueError("Invalid sigma: {}".format(sigma))
        if mc_max_steps_total is not required and mc_max_steps_total < 1:
            raise ValueError("Invalid mc_max_steps_total: {}".format(mc_max_steps_total))
        if mc_ts_per_mc_step is not required and (mc_ts_per_mc_step <= 0) or (mc_ts_per_mc_step > 1):
            raise ValueError("Invalid mc_ts_per_mc_step: {}".format(mc_ts_per_mc_step))

        self.mc_ts_per_mc_step, self.dt, self.m_monte_carlo, self.num_t_samples = self.adjust_mc_ts_per_mc_step(mc_ts_per_mc_step, mc_max_steps_total)

        self.sigma = sigma

        self.t = 0
        self.completed_trajectory = False
        self.started_trajectory = False

        self.max_grad_norm = max_grad_norm

        self.recent_losses = deque(maxlen=10)

        if sigma_rescaling:
            self.sigma_factor = sigma
            self.sqrt_sigma_factor = math.sqrt(sigma)
        else:
            self.sigma_factor = 1.0
            self.sqrt_sigma_factor = 1.0

        # TODO: consider whether to remove these two lines...
        defaults = dict()
        super(PIOMonteCarlo, self).__init__(params, defaults)

    # Minimally adjusts mc_ts_per_mc_step to satisfy 5 constraints:
    #  mc_ts_per_mc_step * m_monte_carlo = 1
    #  mc_max_steps_total = (1/dt) * m_monte_carlo
    #  dt = 1 / int(1/dt)
    #  m_monte_carlo is int
    #  mc_ts_per_mc_step is int

    def adjust_mc_ts_per_mc_step(self, mc_ts_per_mc_step, mc_max_steps_total):
        ideal_m_monte_carlo = 1 / mc_ts_per_mc_step
        # Divisors of mc_max_steps_total
        divisors = [i for i in range(1, mc_max_steps_total + 1) if mc_max_steps_total % i == 0]
        
        # Find the closest divisor
        closest_m_monte_carlo = min(divisors, key=lambda x: abs(x - ideal_m_monte_carlo))
        
        # Adjust mc_ts_per_mc_step
        adjusted_mc_ts_per_mc_step = 1 / closest_m_monte_carlo

        # Adjust dt
        n = mc_max_steps_total // closest_m_monte_carlo
        dt = 1 / n
        
        return adjusted_mc_ts_per_mc_step, dt, closest_m_monte_carlo, n

    def __setstate__(self, state):
        super(PIOMonteCarlo, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def extract_parameters_1d(self):
        """Extracts all parameters into a single 1D array."""
        param_list = []
        
        for group in self.param_groups:
            for p in group['params']:
                #print(f"Parameter: {p}")
                #if p.grad is not None:
                    # Flatten the parameter tensor and add it to the list
                param_list.append(p.data.view(-1))
                #else:
                #    print(f"Parameter has no grad!")

        # Concatenate all parameters into a single 1D tensor
        #print(f"Got param list: {param_list}")
        return torch.cat(param_list)

    def write_1d_params(self, params):
        """Writes a 1D array of parameters back to the model through the optimizer."""
        w_split_index = 0  # Index to track position in the 1D param array
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # Get the number of elements in the parameter tensor
                    param_numel = p.numel()
                    
                    # Extract the relevant portion from the 1D params and reshape
                    new_param_data = params[w_split_index:w_split_index + param_numel].view(p.shape)
                    
                    # Copy the new data back into the parameter tensor
                    p.data.copy_(new_param_data)
                    
                    # Update the split index to move to the next parameter
                    w_split_index += param_numel

    def get_ln_P(self, w):
        self.write_1d_params(w)
        closure = yield
        with torch.enable_grad():
            l = closure()

        self.recent_losses.append(l.item())

        #print(f"Loss: {l.item()}")

        ln_boltz_l = -l / self.sigma
        #print(f"P = exp(-{l.item()} / {sigma}) = {boltz_l.item()}")
        return ln_boltz_l #, loss
    def get_ln_N(self, w):
        n = w.shape[0]
        device = w.device 
        dist = th.distributions.MultivariateNormal(th.zeros(n, device=device), th.eye(n, device=device))  # N(0, I_n)
        #print(f"Getting N probability of w where w min={w.min()}, w max={w.max()}")
        ln_n_prob = dist.log_prob(w)
        #print(f"N = exp(N(0, I_n).log_prob(w)) = {n_prop.item()}")
        return ln_n_prob  # Probability density function
    def get_f(self, w):
        #print(f"Getting f for w = x + math.sqrt(1 - t) * Z = {w}")
        ln_P = yield from self.get_ln_P(w)
        ln_N = self.get_ln_N(w)# * 10000000000000
        ln_f = ln_P - ln_N
        #print(f"get_f: lnP - lnN = {ln_P.item()} - {ln_N.item()} = {ln_f.item()}")
        f = th.exp(ln_f)
        #print(f"get_f: f = exp(ln_f) = {f.item()}")
        return f
    def get_nabla_f(self, f, w):
        #print(f"f.requires_grad: {f.requires_grad}")
        #print(f"w.requires_grad: {w.requires_grad}")

        # todo: consider rewriting using Stein's lemma! avoids grad.
        nabla_f = th.autograd.grad(f, w, create_graph=False)[0]
        # Clip gradients by norm, if max_grad_norm is specified
        #print(f"nabla_f: {nabla_f}")
        if (self.max_grad_norm is not None) and self.max_grad_norm > 0:
            grad_norm = nabla_f.norm()
            if grad_norm > self.max_grad_norm:
                nabla_f = nabla_f * (self.max_grad_norm / grad_norm)

        #print(f"get_nabla_f: nabla_f = {nabla_f}")
        return nabla_f
    def get_drift(self, x: th.Tensor, t):
        #print(f"Getting drift for time t = {t}")
        if not x.requires_grad: # TODO: consider removal
            x.requires_grad = True
        f_sum = 0.0
        nabla_f_sum = None
        for i in range(self.m_monte_carlo):
            #print(f"\nStarting MC round {i}")
            Z = th.randn(x.numel())
            Z = Z.to("cuda")
            w = x + math.sqrt(1 - t) * Z
            if not w.requires_grad: # TODO: consider removal
                w.requires_grad = True
            f = yield from self.get_f(w)
            f_sum += f.item()
            nabla_f = self.get_nabla_f(f, w)
            nabla_f_sum = nabla_f if nabla_f_sum is None else nabla_f_sum + nabla_f
        
            del w, f, nabla_f # free up memory
        #nabla_f_values = [self.get_nabla_f(f, x) for f in f_values]
        nabla_f_sum = nabla_f_sum.to("cuda") # TODO: may not be necessary... or should be earlier

        #print(f"sum_nabla_f: {sum_nabla_f} ")
        #print(f"sum_f: {sum_f.item()} ")

        drift = nabla_f_sum / f_sum
        #print(f"Drift = sum_nabla_f / sum_f = {drift}")
        return drift

    def run_trajectory(self):
        # new_l = yield w

        print(f"STARTING trajectory!")

        # TODO: consider how to run multiple trajectories...
        # or i guess i can handle that later
        # ACTUALLY: we shouldn't do multiple.
        # in pio, traininer != inference. we do multi traj at inf because it's cheap.
        # in mc, training == inference. its too exensive to do multiple.

        initial_w = self.extract_parameters_1d()

        ts = th.linspace(0, 1, self.num_t_samples)
        x = th.zeros(initial_w.numel())
        x = x.to("cuda")
        for i in range(self.num_t_samples):
            self.t = ts[i]
            #print(f"\nStarting time step {i}")
            #if th.isnan(x).any():
            #    print(f"NaN in x! Breaking... x = {x}")
            #    break

            drift = yield from self.get_drift(x, self.t)
            #print(f"Average recent loss = {sum(self.recent_losses) / len(self.recent_losses)}")
            #print("Got drift, continuing")
            drift = drift * self.sigma_factor
            noise = th.randn_like(x) * math.sqrt(self.dt) * self.sqrt_sigma_factor
            noise = noise.to("cuda") # TODO: double check if this is necessary
            x = x + drift * self.dt + noise
            x = th.nan_to_num(x) # same as PIS_NN
            self.write_1d_params(x) # not strictly necessary unless we're evaluating model mid-run
            #print(f"At time {self.t}, x = {x}")
        print(f"Complete! Setting completed_trajectory to True. We found x: {x}")
        self.completed_trajectory = True
        # Final state X_T!
        while True: yield x # TODO: consider replacing with actual closure use

    #@torch.no_grad() # TODO: maybe comment out?
    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """

        if not self.started_trajectory:
            #print(f"self.t=0, so starting trajectory")
            self.trajectory_runner = self.run_trajectory()
            next(self.trajectory_runner)
            self.started_trajectory = True

        # Issue: we can't keep optimizing past t=1 :/
        if not self.completed_trajectory:
            self.trajectory_runner.send(closure)
        else:
            # Unfortunately, lightning still requires us to call closure at each step
            # Unless I find a way to change this behaviour without booting into manual optimization mode
            return closure()

        return self.recent_losses[-1]
        #return loss
