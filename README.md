# Path Integral Optimiser

This repository implements Schrödinger-Föllmer-diffusion-based optimisers, primarily the Path Integral Optimiser, along with several meta-learning tasks. This is for the paper [Path Integral Optimiser: Global Optimisation via Neural Schr\"odinger-F\"ollmer Diffusion](https://github.com/mgm52/Path-Integral-Optimiser/blob/master/OPT_Submission___Path_Integral_Optimiser%20-%20non-anon.pdf), which was accepted to [Neurips OPT 2024](https://opt-ml.org/).

## Setup

The repo heavily depends on [jam](https://github.com/qsh-zh/jam), a versatile toolbox developed by [Qsh.zh](https://github.com/qsh-zh) and [jam-hlt](https://github.com/qsh-zh/jam), a decent deep leanring project template. [⭐️](https://github.com/qsh-zh/jam) if you like them.

*[poetry](https://python-poetry.org/)* (**Recommended**)
```shell
curl -fsS -o /tmp/get-poetry.py https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py
python3 /tmp/get-poetry.py -y --no-modify-path
export PATH=$HOME/.poetry/bin:$PATH
poetry shell
poetry install
```

*pip*
```shell
pip install .
```


## Repo Overview
This repository uses **PyTorch Lightning** to conduct training (e.g. with metrics being Lightning callbacks); **Nevergrad** to run hyperparameter sweeps; **Hydra** to load from config files; and, optionally, **wandb** to log results. It is based on Zhang et al.'s [Path Integral Sampler](https://arxiv.org/abs/2111.15141).

While SGD, Adam, Adagrad and PIS-MC (monte-carlo approximation) are directly applied as optimisers via `train_non_pio.py`, PIO itself has more involved Lightning setup adapted from the original PIS repo, to improve parallelisation. Rather than be treated as an optimiser, PIO is seen as an optimization _target_ whose dataset is the Boltzmann transformation of its task-solving-model's parameters. Below is an overview of how PIO is trained, broken down into sections of PISBasedOptimizer, relied on by `train_pio.py`.

Defining the forward loop, and attachments to data:
```
PISBasedOptimizer
    .model: PIOModel(LightningModule)
        .sde_model: PISNN
            .f()
            .g()
        .training_step(batch) -> loss: float
            # this is where the model is actually run
            # i.e. sdeint_fn runs torchsde.sdeint(sde, y0, ts)
        .configure_optimizers() -> opt: Optimizer
            .task_weights_to_loss()
            .task_next_data() # updates .x, .GT for task_weights_to_loss
```
Performing the Boltzmann transformation:
```
    .datamodule: BasicDataModule
        .dataset: OptGeneral
            .V()    # == .task_weights_to_loss()
            .nll_target_fn()
        .batch_size: int
```
Defining the backwards loop:
```
    .start_train_loop()
        # trainer.fit(self.model, self.datamodule)
            # opt = self.model.configure_optimizers()
            # for data in self.datamodule.next():
                # loss = self.model.training_step()
                # opt.update()
```

### Reproduce

Use run_all_results.py to queue up sweeping & seeding runs.

```
python run_all_results.py
```

This makes calls to `training_coordinator.py` and `viz.py`, which can also be used standalone. e.g.

```
python src/training_coordinator.py experiment=opt_carrillo.yaml model.optimizer=pis-mc model.m_monte_carlo=64 datamodule.dataset.sigma=0.017 model.sde_model.sigma_rescaling=static datamodule.dl.batch_size=1 model.dt=0.05 model.fig_path=/home/max/pis
```

### PIS Reproduce

```
python run.py experiment=ou.yaml logger=wandb
```

See the [folder](configs/experiment) for more experiment configs.

There are some [results](https://wandb.ai/qinsheng/pub_pis?workspace=user-qinsheng) reproduced by the repo.



## Extension Ideas
- Recursive training: train a PIS-based optimizer using a PIS-based optimizer, instead of e.g. Adam?
- Integrate D-Adaptation
  - For PIS learning
  - For finding optimal sigma value?
- Make PIS-Grad viable
  - Use a cheap heuristic grad?
  - Compare using different minibatches per grad computation vs same minibatch
- Methods to improve PIS's stability
  - Other sigma schedules - noisy, to avoid local minima?
  - Gradient clipping
  - Stochastic Weight Averaging?
- Can we initialize PIS to produce weights that correspond with e.g. a Xavier initilisation of task model? Solves issue PIS paper mentions of not knowing good prior to use.
  - Thougthat an advantage of PIS seems to be that it is more resiliant to initialisation? Meaning it works on black box models where good init is unclear anyway.
  - Some work on this in repo already, but incomplete
 
