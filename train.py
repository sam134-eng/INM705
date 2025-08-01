import os
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import VisDroneDataset
from utils import visualize_detections
from logger import Logger
from models import get_model  # moved get_model into models.py

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Object detection collate function
def collate_fn(batch):
    return tuple(zip(*batch))

# Class names for visualization
CLASS_NAMES = [
    'background', 'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

# Training loop
def train_loop(model, train_loader, val_loader, optimizer, num_epochs, logger, lr, backbone):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            running_loss += losses.item()

            if i % 10 == 0:
                print(f"[Epoch {epoch+1} | Batch {i}] Loss: {losses.item():.4f}")
                logger.log_metrics({"train_loss": losses.item()}, step=epoch * len(train_loader) + i)

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")
        logger.log_metrics({"epoch": epoch+1, "avg_train_loss": avg_loss})

        # Save checkpoint with learning rate and epoch
        model_filename = f"fasterrcnn_{backbone}_lr{lr}_ep{epoch+1}.pth"
        torch.save(model.state_dict(), model_filename)

        # Validation preview on one batch
        model.eval()
        with torch.no_grad():
            val_images, val_targets = next(iter(val_loader))
            val_images = [img.to(device) for img in val_images]
            val_targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]
            predictions = model(val_images)

            for img_idx in range(len(val_images)):
                img = val_images[img_idx].cpu()
                target = val_targets[img_idx]
                pred = predictions[img_idx]

                pred_fig = visualize_detections(img, pred['boxes'], pred['labels'], pred['scores'], CLASS_NAMES)
                gt_fig = visualize_detections(img, target['boxes'].cpu(), target['labels'].cpu(),
                                              torch.ones_like(target['labels']), CLASS_NAMES)

                logger.log_image(f"epoch_{epoch+1}_val_pred_{img_idx}", pred_fig)
                logger.log_image(f"epoch_{epoch+1}_val_gt_{img_idx}", gt_fig)

        model.train()

    print("Training complete.")


def main():
    # Hyperparameters
    num_classes = 11  # 10 classes + background
    batch_size = 2
    num_epochs = 30
    lr = 1e-3
    backbone_name = 'resnet50'  # change to 'mobilenet_v3' or 'resnet101'

    # Dataset paths
    train_img_dir = 'VisDrone2019-DET-train/images'
    train_ann_dir = 'VisDrone2019-DET-train/annotations'
    val_img_dir = 'VisDrone2019-DET-val/images'
    val_ann_dir = 'VisDrone2019-DET-val/annotations'

    # Load datasets
    train_dataset = VisDroneDataset(train_img_dir, train_ann_dir, transforms=torchvision.transforms.ToTensor())
    val_dataset = VisDroneDataset(val_img_dir, val_ann_dir, transforms=torchvision.transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Load model and optimizer
    model = get_model(num_classes, backbone_name=backbone_name)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize logger
    logger = Logger(experiment_name=f"visdrone-{backbone_name}-lr{lr}")
    train_loop(model, train_loader, val_loader, optimizer, num_epochs, logger, lr, backbone_name)
    logger.finish()


if __name__ == "__main__":
    main()
