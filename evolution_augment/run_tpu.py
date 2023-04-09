import os
import random
from datetime import datetime

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.experimental.pjrt_backend

from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import wandb

from evolution_augment import config
from evolution_augment import dataloaders
from evolution_augment import models
from evolution_augment import optimizers

os.environ['DATASETS_CACHE_DIR'] = '/home/shumbarw/dataset_cache'

os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'
os.environ['XLA_USE_BF16'] = '1'


def _reduce_fn(x):
    return sum(x) / len(x)


def _train_step(images, labels, model, loss_fn, optimizer, scheduler):
    model.train()
    optimizer.zero_grad()
    logits = model(images)
    loss = loss_fn(logits, labels)
    loss.backward()
    accuracy = (logits.argmax(dim=-1) == labels).float().mean()
    xm.optimizer_step(optimizer, barrier=True)
    scheduler.step()

    global_accuracy = xm.mesh_reduce('accuracy', accuracy, _reduce_fn)
    global_loss = xm.mesh_reduce('global_loss', loss, _reduce_fn)
    return global_loss, global_accuracy


def _train_one_epoch(dataloader, model, loss_fn, optimizer, scheduler,
                     progress_bar, completed_steps):
    device = xm.xla_device()

    losses = []
    accuracies = []
    for step, batch in enumerate(dataloader):
        images = batch['pixel_values']
        labels = batch['labels']
        loss, accuracy = _train_step(images=images,
                                     labels=labels,
                                     model=model,
                                     loss_fn=loss_fn,
                                     optimizer=optimizer,
                                     scheduler=scheduler)
        losses += [loss.detach().item()]
        accuracies += [accuracy.detach().item()]
        mean_loss = _reduce_fn(losses)
        mean_accuracy = _reduce_fn(accuracies)
        if xm.is_master_ordinal():
            step_logs = dict(train_step=completed_steps + step + 1,
                             loss=round(loss.detach().item(), 3),
                             accuracy=round(accuracy.detach().item(), 3),
                             learning_rate=round(scheduler.get_last_lr()[0],
                                                 4))
            progress_bar.set_description(str(step_logs))
            progress_bar.update(1)
    return mean_loss, mean_accuracy


def _validation_step(images, labels, model, loss_fn):
    model.eval()
    logits = model(images)
    loss = loss_fn(logits, labels)
    accuracy = (logits.argmax(dim=-1) == labels).float().mean()
    global_accuracy = xm.mesh_reduce('accuracy', accuracy, _reduce_fn)
    global_loss = xm.mesh_reduce('global_loss', loss, _reduce_fn)
    return global_loss, global_accuracy


def _validate(dataloader, model, loss_fn, progress_bar, train_step):
    device = xm.xla_device()
    losses = []
    accuracies = []
    for step, batch in enumerate(dataloader):
        images = batch['pixel_values']
        labels = batch['labels']
        loss, accuracy = _validation_step(images=images,
                                          labels=labels,
                                          model=model,
                                          loss_fn=loss_fn)
        losses += [loss.detach().item()]
        accuracies += [accuracy.detach().item()]
        mean_loss = _reduce_fn(losses)
        mean_accuracy = _reduce_fn(accuracies)
        if xm.is_master_ordinal():
            step_logs = dict(train_step=train_step,
                             validation_step=step + 1,
                             loss=round(loss.detach().item(), 3),
                             accuracy=round(accuracy.detach().item(), 3))
            progress_bar.update(1)
            progress_bar.set_description(str(step_logs))

    return mean_loss, mean_accuracy


def mp_fn(index, experiment_cfg):
    print('Starting process on rank:', index)
    device = xm.xla_device()

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_seed(42)

    print('Using learning rate:', experiment_cfg.trainers.optimizer.lr)
    model = models.ModelFactory.build(config=experiment_cfg.model).to(device)

    optimizer = optimizers.get_optimizer(
        params=model.parameters(), config=experiment_cfg.trainers.optimizer)
    lr_sched = optimizers.get_lr_scheduler(
        optimizer=optimizer, config=experiment_cfg.trainers.scheduler)

    train_dataloader = dataloaders.DataloaderFactory.build(
        experiment_cfg.train_dataloader)

    mp_train_dataloader = pl.MpDeviceLoader(
        train_dataloader,
        device,
        loader_prefetch_size=16,
        device_prefetch_size=16,
        host_to_device_transfer_threads=8,
    )

    val_dataloader = dataloaders.DataloaderFactory.build(
        experiment_cfg.val_dataloaders)
    mp_val_dataloader = pl.MpDeviceLoader(val_dataloader, device)

    loss_fn = CrossEntropyLoss()

    train_step = 0
    total_train_iterations = experiment_cfg.epochs * len(mp_train_dataloader)

    train_progress_bar = None
    validation_progress_bar = None
    for _epoch in range(experiment_cfg.epochs):
        if xm.is_master_ordinal():
            train_progress_bar = tqdm(initial=train_step,
                                      total=total_train_iterations)
        train_epoch_loss, train_epoch_accuracy = _train_one_epoch(
            dataloader=mp_train_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=lr_sched,
            progress_bar=train_progress_bar,
            completed_steps=train_step)
        train_step += len(mp_train_dataloader)
        if xm.is_master_ordinal():
            train_progress_bar.close()
            epoch_logs = dict(epoch='[{}/{}]'.format(_epoch + 1,
                                                     experiment_cfg.epochs),
                              train_loss=round(train_epoch_loss, 3),
                              train_accuracy=round(train_epoch_accuracy, 3))

        if (_epoch + 1) % 10 == 0:
            if xm.is_master_ordinal():
                validation_progress_bar = tqdm(total=len(mp_val_dataloader))
            with torch.no_grad():
                validation_loss, validation_accuracy = _validate(
                    dataloader=mp_val_dataloader,
                    model=model,
                    loss_fn=loss_fn,
                    progress_bar=validation_progress_bar,
                    train_step=train_step)
            if xm.is_master_ordinal():
                validation_progress_bar.close()
                epoch_logs = dict(**epoch_logs,
                                  validation_loss=round(validation_loss, 3),
                                  validation_accuracy=round(
                                      validation_accuracy, 3))

        if xm.is_master_ordinal():
            print(epoch_logs)


    xm.save(model.state_dict(), './model.pth', master_only=True)


if __name__ == '__main__':
    xmp.spawn(
        fn=mp_fn,
        nprocs=8,
        args=(config.wideresnet28x18_randaugment_cifar10_bs256_ep200_tpu(), ),
        start_method='fork')
