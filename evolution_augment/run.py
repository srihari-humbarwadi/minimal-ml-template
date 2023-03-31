import os
import time
from datetime import datetime

import wandb

from mlproject.boilerplate import Learner
from mlproject.evaluators import ClassificationEvaluator
from mlproject.trainers import ClassificationTrainer

from evolution_augment import config
from evolution_augment import dataloaders
from evolution_augment import models
from evolution_augment import optimizers
from evolution_augment import trainers
from evolution_augment import transforms

if __name__ == '__main__':
    experiment_cfg = config.wideresnet28x18_evoaugment_cifar100_bs256_ep200_n25()

    os.makedirs(experiment_cfg.experiment_dir, exist_ok=True)
    wandb.init(
        name='wideresnet28x18_evoaugment_cifar100_bs256_ep200_n25-[{}]'.format(
            datetime.now().strftime('%d-%m-%Y|%H:%M:%S')),
        project=os.environ.get('WANDB_PROJECT',
                               experiment_cfg.train_dataloader.dataset.name),
        resume='allow',  # allow, True, False, must
        dir=experiment_cfg.experiment_dir,
        save_code=False)

    model = models.ModelFactory.build(config=experiment_cfg.model)

    optimizer = optimizers.get_optimizer(
        params=model.parameters(), config=experiment_cfg.trainers.optimizer)
    lr_sched = optimizers.get_lr_scheduler(
        optimizer=optimizer, config=experiment_cfg.trainers.scheduler)

    train_dataloader = dataloaders.DataloaderFactory.build(
        experiment_cfg.train_dataloader)
    val_dataloader = dataloaders.DataloaderFactory.build(
        experiment_cfg.val_dataloaders)

    if experiment_cfg.train_dataloader.augmentation.evolution_augment:
        evo_aug_cfg = experiment_cfg.train_dataloader.augmentation.evolution_augment
        evolution_augment = transforms.EvolutionAugmentV2(
            num_candidates=evo_aug_cfg.num_candidates,
            num_ops=evo_aug_cfg.num_ops,
            magnitude=evo_aug_cfg.magnitude,
            cutout_length=0,
            mean=experiment_cfg.train_dataloader.dataset.image_means,
            std=experiment_cfg.train_dataloader.dataset.image_stds)

        trainer = trainers.EvolutionAugmentTrainer(
            evolution_augment=evolution_augment,
            optimizer=optimizer,
            scheduler=lr_sched,
            experiment_tracker=wandb)

    else:
        trainer = ClassificationTrainer(
            optimizer=optimizer,
            scheduler=lr_sched,
            experiment_tracker=wandb)

    experiment = Learner(
        experiment_name=experiment_cfg.experiment_name,
        experiment_dir=experiment_cfg.experiment_dir,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloaders=[val_dataloader],
        trainers=[trainer],
        evaluators=[ClassificationEvaluator(experiment_tracker=wandb)],
        evaluate_every_n_steps=250,
        checkpoint_every_n_steps=experiment_cfg.train_iters // 10,
        checkpoint_after_validation=True,
        train_iters=experiment_cfg.train_iters,
        resume=True,
    )
    experiment.run()