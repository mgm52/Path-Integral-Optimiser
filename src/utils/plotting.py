import matplotlib.pyplot as plt

def plot_loss_grads_vs(cfg, losses, maxgrads_preopt, maxgrads_postopt, medianVs, batch_laps):
    # create two plots, one above the other. top shows loss over time, bottom shows maxgrads.
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
    fig.set_size_inches(14, 13)

    for i in range(0, len(losses), batch_laps):
        ax1.axvline(i, color='tab:red', linestyle='--')
        ax2.axvline(i, color='tab:red', linestyle='--')
        ax3.axvline(i, color='tab:red', linestyle='--')
        ax4.axvline(i, color='tab:red', linestyle='--')

    ax1.plot(losses, color='tab:blue')
    ax1.set_title("Loss")

    ax2.plot(maxgrads_preopt, color='tab:green')
    ax2.set_title("Max grad in PIS (pre-optimization)")

    ax3.plot(maxgrads_postopt, color='darkgreen')
    ax3.set_title("Max grad in PIS (post-optimization)")

    ax4.plot(medianVs, color='tab:purple')
    ax4.set_title("Median V(x) (i.e. task loss)")

    plt.suptitle(f"Loss & maxgrad over time (sigma={cfg.datamodule.dataset.sigma}, lr={cfg.model.lr}, trajectories={cfg.datamodule.dl.batch_size},\
                    \nf_format={cfg.model.sde_model.f_format}, gradient_clip_val={cfg.trainer.gradient_clip_val}, gradient_clip_algorithm={cfg.trainer.gradient_clip_algorithm})")
    plt.savefig("loss.png")
    plt.close()