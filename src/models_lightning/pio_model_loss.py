import torch as th

# pylint: disable=too-many-arguments

# Currently, this is the only regularization term (dreg) in use.
# It's given control (i.e. u_t(x)) for dx, added to control*g_coef.
def quad_reg(x, dx, context):
    del x, context
    dx = dx.view(dx.shape[0], -1)
    return 0.5 * dx.pow(2).sum(dim=-1, keepdim=True)


def loss_pis(sdeint_fn, ts, nll_target_fn, nll_prior_fn, y0, n_reg, initial_phase=False, initial_goal=0, initial_target_matching_additive=False):
    # sdeint_fn runs torchsde.sdeint(sde, y0, ts)
        # where y0 is start state, ts is trajectory timestamps
    # ys has shape (len(ts), batch_size, dim)
        # where dim is the state size (i.e. data_ndim + n_reg)
    ys = sdeint_fn(y0, ts)
    # y1 is the terminal state
    y1 = ys[-1]
    dim = y1.shape[1]

    state = th.nan_to_num(y1[:, :-n_reg])
    
    loss = 0
    
    # TODO: could move this line into if statement for performance; currently it is outside for logging
    loss_init = (initial_goal - state).pow(2).mean()
    if initial_phase:
        loss += loss_init
        #print(f"{'initial' if initial_phase else ''} training step: loss_init {loss_init}")
    
    calculate_pis_loss = (not initial_phase) or initial_target_matching_additive
    if calculate_pis_loss:
        reg_loss = y1[:, -n_reg].mean() / dim
        sample_nll = nll_target_fn(state).mean() / dim
        prior_nll = nll_prior_fn(state).mean() / dim
        term_loss = sample_nll - prior_nll
        loss_pis = reg_loss + term_loss
        loss += loss_pis
        #print(f"{'initial' if initial_phase else ''} training step: loss_init {loss_init}, loss_pis {loss_pis}")

    if not calculate_pis_loss:
        return (
            state,
            loss,
            {
                "loss": loss,
                "loss_init": loss_init,
            },
        )
    else:
        return (
            state,
            loss,
            {
                "loss": loss,
                "reg_loss": reg_loss,
                "prior_nll": prior_nll,
                "sample_nll": sample_nll,
                "term_loss": term_loss,
            },
        )