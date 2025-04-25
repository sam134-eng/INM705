import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_fasterrcnn_model(num_classes=91,
                         backbone='resnet50_fpn',
                         pretrained_backbone=True):
    """
    Returns a Faster R-CNN model with customizable backbone and number of classes.

    Args:
        num_classes (int): Number of detection classes (including background).
        backbone (str):  Backbone to use ('resnet50_fpn', 'mobilenet_v2', etc.).
        pretrained_backbone (bool): Whether to use a pretrained backbone.

    Returns:
        torchvision.models.detection.FasterRCNN: The Faster R-CNN model.
    """

    if backbone == 'resnet50_fpn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=pretrained_backbone, num_classes=num_classes
        )
    elif backbone == 'mobilenet_v2':
        backbone = torchvision.models.mobilenet_v2(pretrained=pretrained_backbone).features
        backbone.out_channels = 1280  # MobileNetV2's output channels
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))
        model = FasterRCNN(backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator)
    else:
        raise ValueError(f"Backbone '{backbone}' not supported.")

    return model


if __name__ == '__main__':
        # Example usage
    model = get_fasterrcnn_model(backbone='resnet50_fpn')  # Default
    print("Faster R-CNN with ResNet-50 FPN:")
    print(model)

    model_mobile = get_fasterrcnn_model(backbone='mobilenet_v2')
    print("\nFaster R-CNN with MobileNetV2:")
    print(model_mobile)
