"""
Binary Change Detection Evaluation on WHU-CD and LEVIR-CD

This script implements a standardized evaluation pipeline for binary change
detection tasks, calculating common metrics including mIoU, OA, F1-score, etc.
The ground truth is read directly from binary change labels.

python eval/evaluate_whu_levir.py --pred [PREDICTION_DIR] --gt [GROUND_TRUTH_DIR] --txt [IMAGE_LIST] --threshold [0.5]

"""

import os
from typing import Dict, Optional
import argparse

import cv2
import numpy as np
from tqdm import tqdm



def main():
    """Main execution flow"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Binary Change Detection Evaluation")
    parser.add_argument("--pred", type=str, default="output/WHU/infer_building", help="Prediction directory path")
    parser.add_argument("--gt", type=str, default="<WHU_CD_ROOT>/label_test/", help="Ground truth labels directory")
    parser.add_argument("--txt", type=str, default=None, help="Optional evaluation image list file path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold (0-1 scale)")
    args = parser.parse_args()

    # Initialize metric calculator
    metric = ChangeDetectionMetrics(threshold=args.threshold)

    # Process all images
    if args.txt is None:
        image_list = sorted(os.listdir(args.pred))
    else:
        with open(args.txt, "r") as file:
            image_list = [line.strip() for line in file if line.strip()]

    for filename in tqdm(image_list, desc="Evaluating Predictions"):
        try:
            # Load prediction and binary change label.
            pred_path = os.path.join(args.pred, filename)
            gt_path = os.path.join(args.gt, filename)

            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            # Validate inputs
            if pred is None or gt is None:
                raise ValueError(f"Invalid image file: {filename}")
            if pred.shape != gt.shape:
                raise ValueError(f"Size mismatch in file: {filename}")

            # Update metrics
            metric.update(pred, gt)

        except Exception as e:
            print(f"Skipped {filename}: {str(e)}")
            continue

    # Output results
    results = metric.compute()
    print("\nEvaluation Results:")
    for metric_name, value in results.items():
        print(f"{metric_name:15}: {value:.4f}")

class ChangeDetectionMetrics:
    """
    Change Detection Evaluation Metric Calculator

    Attributes:
        threshold (float): Binarization threshold (0-1 scale)
        eps (float): Numerical stability constant
        tp (float): Accumulated true positives
        tn (float): Accumulated true negatives
        fp (float): Accumulated false positives
        fn (float): Accumulated false negatives
        results (dict): Dictionary storing final evaluation metrics

    Methods:
        reset(): Resets all accumulators
        update(): Updates metric calculations with new batch
        compute(): Computes and returns all metrics
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Initialize metric calculator
        
        Args:
            threshold: Binarization threshold (0-1 scale), default 0.5
        """
        self.threshold = threshold * 255.0  # Convert to pixel value
        self.eps = 1e-7  # Numerical stability constant
        
        # Initialize accumulators
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0
        
        # Store final results
        self.results: Optional[Dict[str, float]] = None

    def reset(self) -> None:
        """Resets all accumulators to zero"""
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        """
        Update metrics with new data pair
        
        Args:
            prediction: Model prediction (grayscale image, 0-255)
            target: Ground truth (grayscale image, 0-255)
        """
        # Convert to binary masks
        pred_binary = (prediction > self.threshold)
        target_binary = (target > self.threshold)

        # Update confusion matrix elements
        self.tp += np.sum(pred_binary & target_binary)
        self.tn += np.sum(~pred_binary & ~target_binary)
        self.fp += np.sum(pred_binary & ~target_binary)
        self.fn += np.sum(~pred_binary & target_binary)

    def compute(self) -> Dict[str, float]:
        """Compute and return all evaluation metrics"""
        # Calculate IoU for both classes
        iou_change = self.tp / (self.tp + self.fp + self.fn + self.eps)
        iou_nochange = self.tn / (self.tn + self.fp + self.fn + self.eps)
        
        # Calculate mean IoU
        miou = 0.5 * (iou_change + iou_nochange)
        
        # Calculate overall accuracy
        oa = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + self.eps)
        
        # Calculate precision/recall/F1-score
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall = self.tp / (self.tp + self.fn + self.eps)
        f1_score = (2 * precision * recall) / (precision + recall + self.eps)

        # Organize results
        self.results = {
            'miou': miou,
            'oa': oa,
            'iou_change': iou_change,
            'iou_nochange': iou_nochange,
            'f1_score_change': f1_score,
            'precision_change': precision,
            'recall_change': recall
        }
        return self.results
    
if __name__ == "__main__":
    main()
