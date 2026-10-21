"""torchvision.models stub: only pulled in by torchmetrics image extras at import time."""
class _Weights:
    IMAGENET1K_V1 = None
    DEFAULT = None


def _missing(*a, **k):
    raise NotImplementedError("torchvision.models stub")


VGG16_Weights = _Weights
VGG19_Weights = _Weights
ResNet50_Weights = _Weights
for _n in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
           "resnext50_32x4d", "wide_resnet50_2", "mobilenet_v2", "efficientnet_b0",
           "vgg16", "vgg19", "alexnet", "squeezenet1_0", "densenet121"]:
    globals()[_n] = _missing
