"""
Change Detection (SECOND dataset) Evaluation Metric Calculation

This script implements a standardized evaluation pipeline for change detection tasks,
calculating common metrics including mIoU, OA, F1-score, etc.

python eval/evaluate_second.py --pred [PREDICTION_DIR] --gt1 [GROUND_TRUTH_DIR1] --gt2 [GROUND_TRUTH_DIR2] --class-name [CLASS_NAME] --threshold [0.5]

"""

import os
from typing import Dict, Optional
import argparse

import cv2
import numpy as np
from tqdm import tqdm



def main():
    """Main execution flow"""
    # Configuration settings
    CLASS_MAPPING = {
        'background': 0,
        'water': 1,  # class_0
        'ground': 2,  # class_1
        'low vegetation': 3,  # class_2
        'tree': 4,  # class_3
        'building': 5,  # class_4
        'playground': 6,  # class_5
    }

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Semantic Change Detection Evaluation')
    parser.add_argument('--pred', type=str, default="output/SECOND/infer_playground", help='Prediction directory path')
    parser.add_argument('--gt1', type=str, default="<SECOND_ROOT>/val/label1", help='First time point labels directory')
    parser.add_argument('--gt2', type=str, default="<SECOND_ROOT>/val/label2", help='Second time point labels directory')
    parser.add_argument('--class-name', type=str, default='playground',
                       choices=list(CLASS_MAPPING.keys()), help='Target class for change detection')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Binarization threshold (0-1 scale)')
    args = parser.parse_args()

    # Initialize metric calculator
    metric = ChangeDetectionMetrics(threshold=args.threshold)
    class_id = CLASS_MAPPING[args.class_name]

    # Process all images
    # with open(args.txt, 'r') as f:
    #     image_list = [line.strip() for line in f if line.strip()]
    image_list = sorted(os.listdir(args.pred))
    for filename in tqdm(image_list, desc='Evaluating Predictions'):
        try:
            # Load data
            pred_path = os.path.join(args.pred, filename)
            label1_path = os.path.join(args.gt1, filename)
            label2_path = os.path.join(args.gt2, filename)

            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            label1 = cv2.imread(label1_path, cv2.IMREAD_ANYCOLOR)
            label2 = cv2.imread(label2_path, cv2.IMREAD_ANYCOLOR)

            # Validate inputs
            if pred.shape != label1.shape or pred.shape != label2.shape:
                raise ValueError(f"Size mismatch in file: {filename}")

            # Generate change label
            change_label = ((label1 == class_id) | (label2 == class_id)).astype(np.uint8) * 255
            
            # Update metrics
            metric.update(pred, change_label)

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
