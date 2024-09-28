# Path Integral Optimiser

This repository implements a diffusion-based optimizer using PyTorch Lightning and a custom neural network called PISNN. This repository is largely an adaptation of Zhang et al.'s [Path Integral Sampler](https://arxiv.org/abs/2111.15141), the setup for which remains the same.

While SGD, Adam, Adagrad and PIS-MC (monte-carlo approximation) are more straightforwardly trained through optimisers in pis_optim_pl.py, PIO itself has a rather complicated setup adapted from the original PIS repo. Rather than be treated as an optimiser, PIO is seen as an optimization target for which the dataset is the Boltzmann transformation of its task-solving-model's parameters. Below is an overview of how PIO is trained, broken down into sections of PISBasedOptimizer, relied on by pis_optim_pl.py.

Defining the forward loop, and attachments to data:
```
PISBasedOptimizer
    .model: BaseModel(LightningModule)
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
    .datamodule: MyDataModule
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

## Reproduce

Use run_all_results.py to queue up sweeping & seeding runs. Right now this is left as a hardcoded script, with choices in `experiments` and `optimizers`.

```
python run_all_results.py
```


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
 
## Further PIS Details

### PIS Reproduce

```
python run.py experiment=ou.yaml logger=wandb
```

See the [folder](configs/experiment) for more experiment configs.

There are some [results](https://wandb.ai/qinsheng/pub_pis?workspace=user-qinsheng) reproduced by the repo.

### PIS Reference

```tex
@inproceedings{zhang2021path,
  author    = {Qinsheng Zhang and Yongxin Chen},
  title     = {Path Integral Sampler: a stochastic control approach for sampling},
  booktitle = {International Conference on Learning Representations},
  year      = {2022}
}
```
