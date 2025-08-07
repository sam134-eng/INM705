import os
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import VisDroneDataset
from models import get_model
from torchvision.ops import box_iou
import numpy as np
import pandas as pd
import re

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_iou(box1, box2):
    box1 = torch.tensor(box1).unsqueeze(0)
    box2 = torch.tensor(box2).unsqueeze(0)
    iou = box_iou(box1, box2)
    return iou.item()


def compute_ap(recalls, precisions):
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap


def calculate_map(preds, gts, iou_threshold):
    all_ap = []
    for (pred_boxes, pred_labels, pred_scores), (gt_boxes, gt_labels) in zip(preds, gts):
        if len(pred_scores) == 0:
            all_ap.append(0.0)
            continue

        matched_gt = set()
        tp, fp = [], []
        sorted_indices = np.argsort(-np.array(pred_scores))
        pred_boxes = [pred_boxes[i] for i in sorted_indices]
        pred_labels = [pred_labels[i] for i in sorted_indices]

        for pb, pl in zip(pred_boxes, pred_labels):
            match_found = False
            for idx, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                if idx in matched_gt or pl != gl:
                    continue
                iou = compute_iou(pb, gb)
                if iou >= iou_threshold:
                    tp.append(1)
                    fp.append(0)
                    matched_gt.add(idx)
                    match_found = True
                    break
            if not match_found:
                tp.append(0)
                fp.append(1)

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        denom = len(gt_boxes) if len(gt_boxes) > 0 else 1
        recalls = tp_cumsum / denom
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        ap = compute_ap(recalls, precisions)
        all_ap.append(ap)

    return np.mean(all_ap)


def evaluate(model, dataloader, iou_thresholds):
    model.eval()
    preds_all, gts_all = [], []
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for i, output in enumerate(outputs):
                boxes = output['boxes'].cpu().tolist()
                scores = output['scores'].cpu().tolist()
                labels = output['labels'].cpu().tolist()
                gt_boxes = targets[i]['boxes'].cpu().tolist()
                gt_labels = targets[i]['labels'].cpu().tolist()
                preds_all.append((boxes, labels, scores))
                gts_all.append((gt_boxes, gt_labels))

    results = {}
    for iou in iou_thresholds:
        mAP = calculate_map(preds_all, gts_all, iou_threshold=iou)
        results[f"iou_{int(iou * 100)}"] = round(mAP, 4)
        print(f"mAP @ IoU={iou:.2f}: {mAP:.4f}")
    return results


def extract_lr_epoch(model_path):
    match = re.search(r"lr([\d.]+)_ep(\d+)", model_path)
    if match:
        lr, ep = match.groups()
        return float(lr), int(ep)
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--ann_dir', type=str, required=True)
    parser.add_argument('--models_dir', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--csv_path', type=str, default='results_log.csv')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--iou_thresholds', type=float, nargs='+', default=[0.5, 0.75])
    args = parser.parse_args()

    dataset = VisDroneDataset(args.img_dir, args.ann_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda x: tuple(zip(*x)))

    rows = []
    for file in os.listdir(args.models_dir):
        if file.endswith(".pth"):
            model_path = os.path.join(args.models_dir, file)
            lr, ep = extract_lr_epoch(file)
            model = get_model(num_classes=11, backbone_name=args.backbone)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            print(f"\nEvaluating: {file}")
            results = evaluate(model, dataloader, iou_thresholds=args.iou_thresholds)
            row = {"model_path": file, "backbone": args.backbone, "lr": lr, "epochs": ep, **results}
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.csv_path, index=False)
    print(f"\n✅ Results saved to {args.csv_path}")


if __name__ == "__main__":
    main()
