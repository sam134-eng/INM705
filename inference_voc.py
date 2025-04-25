import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision import transforms
from models import get_fasterrcnn_model  # Import the model
from utils import load_config

def load_model(model_path, config):
    num_classes = config['model']['num_classes']
    model = get_fasterrcnn_model(num_classes=num_classes,
                                 backbone='resnet50_fpn',
                                 pretrained_backbone=False)  # Important: pretrained=False
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def inference_faster_rcnn(img_path, model, class_names):
    image = Image.open(img_path).convert("RGB")
    img_tensor = transforms.ToTensor()(image)
    with torch.no_grad():
        output = model([img_tensor])[0]
    # Draw boxes
    plt.imshow(image)
    ax = plt.gca()
    boxes = output['boxes']
    labels = output['labels']
    scores = output['scores']
    for i in range(len(boxes)):
        if scores[i] > 0.5:
            box = boxes[i].detach().numpy()
            label_idx = labels[i].item()
            class_name = class_names[label_idx]
            rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                             linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(box[0], box[1] - 10, f"{class_name}: {scores[i]:.2f}",
                    color='red', fontsize=8)
    plt.axis('off')
    plt.show()
    return labels.tolist(), boxes.tolist(), scores.tolist()


if __name__ == '__main__':
    config = load_config('config.yaml')
    model_path = 'fasterrcnn_epoch_10.pth'
    model = load_model(model_path, config)
    coco_class_names = [
        'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
        'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
        'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    test_images = ['test_images/cat.jpg', 'test_images/bike_2.jpg',
                   'test_images/snowboard.jpg']
    for img_path in test_images:
        print(f"\n⏱ Inference on {img_path}")
        labels, boxes, scores = inference_faster_rcnn(img_path, model,
                                                       coco_class_names)
        print("Predicted labels:", labels)
        print("Predicted boxes:", boxes)
        print("Predicted scores:", scores)

