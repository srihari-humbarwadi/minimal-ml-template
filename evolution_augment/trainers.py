from accelerate import Accelerator

from evolution_augment.transforms import EvolutionAugmentV2
from mlproject.decorators import collect_metrics
from mlproject.trainers import ClassificationTrainer, TrainerOutput


class EvolutionAugmentTrainer(ClassificationTrainer):
    def __init__(self, evolution_augment: EvolutionAugmentV2, **kwargs):
        self.evolution_augment = evolution_augment
        super(EvolutionAugmentTrainer, self).__init__(**kwargs)


    @collect_metrics
    def training_step(
        self,
        model,
        batch,
        batch_idx,
        step_idx,
        epoch_idx,
        accelerator: Accelerator,
    ) -> TrainerOutput:
        
        batch['pixel_values'] = self.evolution_augment(
            model, batch['pixel_values'], batch['labels'])
        
        return super(EvolutionAugmentTrainer, self).training_step(
            model=model,
            batch=batch,
            batch_idx=batch_idx,
            step_idx=step_idx,
            epoch_idx=epoch_idx,
            accelerator=accelerator)