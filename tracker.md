TODO:

1. to change victim to `SwinTransformer`, using `SimpleCNN3D` in `models/simple_models.py` for now
2. to load pretrained weigths for victim
3. to change test dataset, using `RandomVideosLikeKinetics` in `datasets/datasets.py` for now, to use `Kinetics400` (to change `datasets/datasets.py` L87-99, and in cmd line arg)
4. to change generators and discriminator, for now using `SimpleGenerator`, `SimpleDiscriminator` in `models/simple_models.py`
5. to change adversary, for now using `SimpleCNN3D` in `models/simple_models.py`
6. to verify the working of changed `zoge_backward()` in `attacks/attack_utils.py`