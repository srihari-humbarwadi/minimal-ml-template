import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F

class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class EvolutionAugmentV2:
    def __init__(self, num_candidates, num_ops, magnitude, cutout_length, mean, std):
        self.num_candidates = num_candidates
        self.num_ops = num_ops,
        self.magnitude = magnitude
        self.cutout_length = cutout_length
        
        self.means = mean
        self.stds = std
        
        self._rand_augment = transforms.RandAugment(
            num_ops=num_ops, magnitude=magnitude)
        self._cutout = CutoutDefault(length=cutout_length)
        self._normlize_fn = transforms.Normalize(mean=mean, std=std)
        
    def _get_greedy_batch(self, model, images, labels):
        model.eval()
        greedy_batch = images.detach().clone()
        greedy_losses = torch.zeros_like(labels, dtype=torch.float)

        for _ in range(self.num_candidates):
            augmented_images = self._rand_augment((images * 255).to(torch.uint8)).to(images.dtype) / 255.0
            logits = model(augmented_images)
            loss = F.cross_entropy(logits, labels, reduction='none').detach()
            greedy_batch = torch.where(
                loss.view(-1, 1, 1, 1) > greedy_losses.view(-1, 1, 1, 1),
                augmented_images, greedy_batch)
            greedy_losses = torch.where(
                loss > greedy_losses,
                loss, greedy_losses)
        return greedy_batch
    
    def __call__(self, model, images, labels):
        greedy_batch = self._get_greedy_batch(model, images, labels)
        return self._normlize_fn(greedy_batch)

class RandAugmentV2(transforms.RandAugment):
    _SUPPORTED_OPS = {
        'Identity', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
        'Brightness', 'Color', 'Contrast', 'Sharpness', 'Posterize',
        'Solarize', 'AutoContrast', 'Equalize'
    }

    def __init__(self,
                 op_list,
                 num_ops,
                 magnitude,
                 num_magnitude_bins,
                 interpolation=transforms.InterpolationMode.NEAREST,
                 fill=None):
        super(RandAugmentV2,
              self).__init__(num_ops=num_ops,
                             magnitude=magnitude,
                             num_magnitude_bins=num_magnitude_bins,
                             interpolation=interpolation,
                             fill=fill)
        if isinstance(op_list, (list, tuple)):
            self.op_list = set(op_list)
        else:
            raise ValueError(
                'Expected `op_list` to be of type `{{list, tuple}}` but got {}'
                .format(type(op_list)))
        for op in op_list:
            if op not in RandAugmentV2._SUPPORTED_OPS:
                raise ValueError(
                    'Got unsupport OP type, supported OPs are: {}'.format(
                        RandAugmentV2._SUPPORTED_OPS))

    def _augmentation_space(self, num_bins, image_size):
        augmentation_space = super(RandAugmentV2, self)._augmentation_space(
            num_bins, image_size)
        return {
            k: v
            for k, v in augmentation_space.items() if k in self.op_list
        }


class EvolutionAugment:

    def __init__(self, augmentation_ops, num_candidates, min_augmentations,
                 max_augmentations, max_magnitude, num_magnitude_bins,
                 interpolation, normalize=True, image_means=None, image_stds=None):
        self.augmentation_ops = augmentation_ops
        self.num_candidates = num_candidates
        self.min_augmentations = min_augmentations
        self.max_augmentations = max_augmentations
        self.max_magnitude = max_magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.normalize = normalize
        self.image_means = image_means
        self.image_stds = image_stds
        self._to_tensor = transforms.ToTensor()

        if normalize:
            if image_means is None:
                raise ValueError('`image_means` cannot be None when `normalize=True`')

            if image_stds is None:
                raise ValueError('`image_stds` cannot be None when `normalize=True`')

            self.normalize_fn = transforms.Normalize(
                mean=image_means, std=image_stds)


    def _generate_candidate(self):
        num_ops = torch.randint(low=self.min_augmentations,
                                high=self.max_augmentations + 1,
                                size=(1, )).item()
        magnitude = torch.randint(low=0, high=self.max_magnitude, size=(1, ))
        candidate = RandAugmentV2(op_list=self.augmentation_ops,
                                  num_ops=num_ops,
                                  magnitude=int(magnitude),
                                  num_magnitude_bins=self.num_magnitude_bins)
        return (candidate, magnitude)

    def _generate_population(self):
        population = []
        for _ in range(self.num_candidates):
            population += [self._generate_candidate()]

        return population

    def __call__(self, sample):
        population = self._generate_population()
        augmented_images = []
        magnitudes = []
        for candidate, magnitude in population:
            augmented_image = self._to_tensor(candidate(sample['img'][0]))
            augmented_images += [self.normalize_fn(augmented_image)]
            magnitudes += [magnitude]

        sample['augmented_images'] = [torch.stack(augmented_images, dim=0)]
        sample['magnitudes'] = [torch.stack(magnitudes, dim=0)]
        return sample
