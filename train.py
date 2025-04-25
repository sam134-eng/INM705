import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.optim as optim
from dataset import COCODetectionDataset
from pycocotools.coco import COCO
from utils import visualize_detections # Import visualize_detections
from logger import Logger # Import Logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def train_loop(model, train_loader, val_loader, optimizer, num_epochs, logger): # Add val_loader and logger
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, (images, targets) in enumerate(train_loader):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

            if i % 10 == 0:
                print(f"[Epoch {epoch+1} | Batch {i}] Loss: {losses.item():.4f}")
                logger.logger.log({"train_loss": losses.item()})

        print(f"Epoch {epoch+1} complete. Avg Loss: {running_loss/len(train_loader):.4f}")
        logger.logger.log({"epoch": epoch+1, "avg_train_loss": running_loss/len(train_loader)})
        torch.save(model.state_dict(), f"fasterrcnn_epoch_{epoch+1}.pth")

        # Visualize on validation set
        model.eval()
        with torch.no_grad():
            val_images, val_targets = next(iter(val_loader)) # Get a batch of validation data
            val_images = list(img.to(device) for img in val_images)
            val_targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]
            
            predictions = model(val_images)
            
            for img_idx in range(len(val_images)): # Loop through each image in the batch.
                img = val_images[img_idx].cpu()
                target = val_targets[img_idx]
                pred = predictions[img_idx]
                
                # Convert target boxes to the same format as predicted boxes
                target_boxes = target['boxes'].cpu()
                target_labels = target['labels'].cpu()
                
                class_names = ['background', 'person', 'bicycle', 'car', 'motorcycle',
                               'airplane', 'bus', 'train', 'truck', 'boat',
                               'traffic light', 'fire hydrant', 'stop sign',
                               'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                               'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                               'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                               'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                               'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
                
                # Visualize predictions
                pred_img = visualize_detections(img, pred['boxes'], pred['labels'], pred['scores'], class_names=class_names, box_format='xyxy')
                
                # Visualize ground truth
                gt_img = visualize_detections(img, target_boxes, target_labels, torch.ones_like(target_labels), class_names=class_names, box_format='xyxy') #scores are dummy
                
                # Log the images to W&B
                logger.logger.log({
                    f"epoch_{epoch}_val_pred_{img_idx}": wandb.Image(pred_img, caption=f"Epoch {epoch} - Prediction {img_idx}"),
                    f"epoch_{epoch}_val_gt_{img_idx}": wandb.Image(gt_img, caption=f"Epoch {epoch} - Ground Truth {img_idx}")
                })
        model.train() #set back to train after validation
    print("Training complete.")

def main():
    num_classes = 91  # 80 COCO classes + 1 background

    # Dataset paths
    train_img_dir = 'datasets/coco/train2017'
    train_ann_file = 'datasets/coco/annotations/instances_train2017.json'
    val_img_dir = 'datasets/coco/val2017'  # Add validation data path
    val_ann_file = 'datasets/coco/annotations/instances_val2017.json'

    # Hyperparameters
    batch_size = 2
    num_epochs = 10
    lr = 1e-4

    # Load dataset
    train_dataset = COCODetectionDataset(train_img_dir, train_ann_file)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    val_dataset = COCODetectionDataset(val_img_dir, val_ann_file) #create validation dataset
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn) #create validation dataloader

    # Model and optimizer
    model = get_model(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize logger
    logger = Logger("faster_rcnn_training")

    # Train the model
    train_loop(model, train_loader, val_loader, optimizer, num_epochs, logger) # Pass val_loader and logger

    logger.logger.finish()

if __name__ == "__main__":
    main()