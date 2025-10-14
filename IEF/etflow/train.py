import argparse
import os.path as osp

import torch
from lightning.pytorch import seed_everything
from loguru import logger as log

from etflow.utils import (
    instantiate_callbacks,
    instantiate_datamodule,
    instantiate_logger,
    instantiate_model,
    instantiate_trainer,
    log_hyperparameters,
    read_yaml,
    setup_log_dir,
)

torch.set_float32_matmul_precision("high")


def run(config: dict) -> None:

    # seed everything for reproducibility
    seed_everything(config.get("seed", 42))

    # task name for logger, if not provided use default
    task_name = config.get("task_name", None)
    
    if config['train_mode'] == 'baseline':
        task_name = (
            f"{config['train_mode']}-"
            f"{task_name}-sigma{config['model_args']['sigma']}"
            f"{config['model_args']['prior_type']}"
        )
    else:
        task_name = (
            f"{config['train_mode']}-z{config['model_args']['z']}-"
            f"{config['max_perms']}perms-"
            f"{config['num_rotations']}rots-"
            f"{task_name}-sigma{config['model_args']['sigma']}"
            f"{config['model_args']['prior_type']}"
        )

    print(task_name)
    # instantiate logger (skip if debug mode)
    logger = None
    if config.get("logger") is not None:
        logger = instantiate_logger(
            config.get("logger", "default_logger"),
            config.get("logger_args") or {},
            task_name=task_name,
        )

    # setup log directory
    setup_log_dir(task_name)

    # instantiate datamodule
    datamodule = instantiate_datamodule(config["datamodule"], config["datamodule_args"], 
                                        train_mode=config["train_mode"], 
                                        max_perms=config["max_perms"],
                                        )

    # instantiate model
    model = instantiate_model(config["model"], config["model_args"], config["train_mode"], config["num_rotations"])
    # exit()
    pretrained_ckpt = config.get("pretrained_ckpt", None)
    if pretrained_ckpt is not None:
        assert osp.exists(
            pretrained_ckpt
        ), f"Pretrained checkpoint {pretrained_ckpt} not found!"
        state_dict = torch.load(pretrained_ckpt, map_location=model.device)[
            "state_dict"
        ]
        model.load_state_dict(state_dict, strict=False)
        log.info(f"Loaded pretrained model from checkpoint: {pretrained_ckpt}")

    # instantiate callbacks
    callbacks = instantiate_callbacks(config["callbacks"])

    # instantiate trainer
    trainer = instantiate_trainer(
        config["trainer"],
        config["trainer_args"],
        logger=logger,
        callbacks=callbacks,
    )

    # log config
    log_hyperparameters({"cfg": config, "model": model, "trainer": trainer})

    # start training
    resume_ckpt_path = config.get("ckpt_path", None)
    if resume_ckpt_path is not None:
        print(f"Resuming training from checkpoint: {resume_ckpt_path}")

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt_path)


if __name__ == "__main__":
    # read config path
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--train_mode", type=str, default="baseline")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    parser.add_argument("--num_rotations", type=int, default=1)
    parser.add_argument("--max_perms", type=int, default=50)
    parser.add_argument("--z", type=float, default=0.4)
    parser.add_argument("--sigma", type=float, default=0.1)
    args = parser.parse_args()

    # read config
    osp.exists(args.config), f"Config file {args.config} not found"
    config = read_yaml(args.config)

    # update config with debug mode
    config["train_mode"] = args.train_mode
    
    if args.resume_path is not None:
        config["ckpt_path"] = args.resume_path
        
    if args.pretrained_ckpt is not None:
        config["pretrained_ckpt"] = args.pretrained_ckpt
        
    config["num_rotations"] = args.num_rotations
    config["max_perms"] = args.max_perms
    config["model_args"]["z"] = args.z
    config["model_args"]["sigma"] = args.sigma
        
    run(config)
