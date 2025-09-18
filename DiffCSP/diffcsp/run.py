from pathlib import Path
from typing import List
import sys
sys.path.append('.')
import hydra
import numpy as np
import torch
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

from diffcsp.common.utils import log_hyperparameters, PROJECT_ROOT

import wandb
import copy
from pytorch_lightning.callbacks import ModelCheckpoint

class SmoothedTrendEarlyStopping(pl.Callback):
    def __init__(self, patience=5, min_epochs=10, trend_epochs=5, smooth_epochs=5, delta=0.001, verbose=False):
        super().__init__()
        self.patience = patience
        self.min_epochs = min_epochs
        self.trend_epochs = trend_epochs
        self.smooth_epochs = smooth_epochs
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.stopped_epoch = 0
        self.epochs_since_improvement = 0
        self.smoothed_losses = []

    def on_validation_end(self, trainer, pl_module):
        # Check if the number of epochs is enough to evaluate trend
        if trainer.current_epoch >= self.min_epochs:
            val_loss = trainer.callback_metrics['val_loss'].item()

            # Smooth the last `smooth_epochs` losses
            self.smoothed_losses.append(val_loss)
            if len(self.smoothed_losses) > self.smooth_epochs:
                self.smoothed_losses.pop(0)

            # Calculate smoothed trend (mean of the last `smooth_epochs`)
            smoothed_loss = sum(self.smoothed_losses) / len(self.smoothed_losses)

            # Check if the loss trend is improving
            if self.best_loss is None or smoothed_loss < self.best_loss - self.delta:
                self.best_loss = smoothed_loss
                self.epochs_since_improvement = 0
            else:
                self.epochs_since_improvement += 1

            # Early stop if no improvement after `patience` epochs
            if self.epochs_since_improvement >= self.patience:
                if self.verbose:
                    print(f"Epoch {trainer.current_epoch}: Early stopping triggered.")
                trainer.should_stop = True


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info("Adding callback <ModelCheckpoint - Best and Last>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(HydraConfig.get().run.dir),
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
                save_last=cfg.train.model_checkpoints.save_last,
                filename="{epoch}-{val_loss:.4f}",  # Optional: customize naming
            )
        )

        hydra.utils.log.info("Adding callback <ModelCheckpoint - Every N Epochs>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(HydraConfig.get().run.dir),
                every_n_epochs=cfg.train.model_checkpoints.every_n_epochs,
                save_top_k=-1,  # Save all
                save_last=False,
                monitor=None,   # No monitoring
                filename="epoch{epoch:03d}",  # Optional: consistent naming
            )
        )

    return callbacks


def run(cfg: DictConfig) -> None:
    """
    Generic train loop

    :param cfg: run configuration, defined by Hydra in /conf
    """
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run=}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.pl_trainer.gpus = 0
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0

        # Switch wandb mode to offline to prevent online logging
        cfg.logging.wandb.mode = "offline"

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)
    hydra_dir.mkdir(parents=True, exist_ok=True)  # Create if missing

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    # Pass scaler from datamodule to model
    hydra.utils.log.info(f"Passing scaler from datamodule to model <{datamodule.scaler}>")
    if datamodule.scaler is not None:
        model.lattice_scaler = datamodule.lattice_scaler.copy()
        model.scaler = datamodule.scaler.copy()
    torch.save(datamodule.lattice_scaler, hydra_dir / 'lattice_scaler.pt')
    torch.save(datamodule.scaler, hydra_dir / 'prop_scaler.pt')
        
    if hasattr(cfg.train, "resume_path"):
        state_dict = torch.load(cfg.train.resume_path, map_location=model.device)[
            "state_dict"
        ]
        model.load_state_dict(state_dict, strict=False)
        
    if cfg.train.adaptive_permute:
        datamodule.set_schedulers(model.beta_scheduler, model.sigma_scheduler)
    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    # Logger instantiation/configuration
    wandb_logger = None
    if "wandb" in cfg.logging:
        hydra.utils.log.info("Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            **wandb_config,
            settings=wandb.Settings(start_method="fork"),
            tags=cfg.core.tags,
        )
        hydra.utils.log.info("W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )

    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)

    # Load checkpoint (if exist)
    ckpt = None
    ckpts = list(hydra_dir.glob('*.ckpt'))
    if len(ckpts) == 0:
        if hasattr(cfg.train, "real_resume_path"):
            ckpts = list(Path(cfg.train.real_resume_path).glob('*.ckpt'))
    if len(ckpts) > 0:
        for ck in ckpts:
            if 'last' in ck.parts[-1]:
                ckpt = str(ck)
        if ckpt is None:
            ckpt_epochs = np.array(
                [int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts if 'last' not in ckpt.parts[-1]])
            ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
                
    hydra.utils.log.info(f"found checkpoint: {ckpt}")
    print(f"found checkpoint: {ckpt}")
                
    hydra.utils.log.info("Instantiating the Trainer")

    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        # resume_from_checkpoint=ckpt,
        **cfg.train.pl_trainer,
    )

    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    hydra.utils.log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt)

    hydra.utils.log.info("Starting testing!")
    trainer.test(datamodule=datamodule)

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
