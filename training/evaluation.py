"""
Model Evaluation and Quality Assurance Pipeline
Comprehensive evaluation of trained YOLOv8 models with detailed metrics and visualizations.

Features:
- Performance metrics calculation (mAP, precision, recall)
- Per-class analysis
- Confusion matrix generation
- Regional object performance analysis
- Visualization dashboard
- Model comparison tools
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
import torch
from PIL import Image
import cv2
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import logging
from datetime import datetime
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path, data_yaml_path, output_dir="evaluation_results"):
        self.model_path = Path(model_path)
        self.data_yaml_path = Path(data_yaml_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load model and data config
        self.model = YOLO(str(self.model_path))
        
        with open(self.data_yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        self.class_names = self.data_config['names']
        self.num_classes = len(self.class_names)
        
        # Results storage
        self.evaluation_results = {}
        self.predictions = []
        self.ground_truths = []
        
        # Pakistan-specific classes for regional analysis
        self.pakistan_classes = [
            'pakistani_rupee_notes', 'rupee_coins', 'roti_chapati', 'naan_bread',
            'biryani_rice', 'chai_tea_cup', 'shalwar_kameez', 'dupatta_scarf',
            'rickshaw_auto', 'chingchi_rickshaw', 'prayer_mat_janamaz', 'tasbih_beads'
        ]
        
    def run_official_evaluation(self):
        """Run official YOLOv8 evaluation"""
        logger.info("Running official YOLOv8 evaluation...")
        
        try:
            # Run validation on test set
            results = self.model.val(
                data=str(self.data_yaml_path),
                split='test',
                save_txt=True,
                save_conf=True,
                plots=True,
                verbose=True
            )
            
            # Extract metrics
            metrics = {
                'mAP50': float(results.box.map50),
                'mAP50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'fitness': float(results.fitness)
            }
            
            # Per-class metrics
            if hasattr(results.box, 'maps'):
                per_class_map = results.box.maps.tolist()
                metrics['per_class_mAP'] = {
                    self.class_names[i]: per_class_map[i] 
                    for i in range(min(len(per_class_map), len(self.class_names)))
                }
            
            self.evaluation_results['official_metrics'] = metrics
            logger.info(f"Official evaluation completed. mAP50: {metrics['mAP50']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Official evaluation failed: {e}")
            return {}
    
    def collect_predictions_and_ground_truth(self, test_images_dir, test_labels_dir):
        """Collect predictions and ground truth for detailed analysis"""
        logger.info("Collecting predictions and ground truth...")
        
        test_images_dir = Path(test_images_dir)
        test_labels_dir = Path(test_labels_dir)
        
        image_files = list(test_images_dir.glob('*.jpg')) + list(test_images_dir.glob('*.png'))
        
        for img_path in tqdm(image_files, desc="Processing test images"):
            try:
                # Load image
                image = Image.open(img_path)
                img_width, img_height = image.size
                
                # Get predictions
                results = self.model(image, verbose=False)[0]
                
                # Process predictions
                pred_boxes = []
                pred_classes = []
                pred_scores = []
                
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    
                    pred_boxes.append([x1, y1, x2, y2])
                    pred_classes.append(class_id)
                    pred_scores.append(confidence)
                
                # Load ground truth
                label_path = test_labels_dir / f"{img_path.stem}.txt"
                gt_boxes = []
                gt_classes = []
                
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                center_x, center_y, width, height = map(float, parts[1:5])
                                
                                # Convert to absolute coordinates
                                x1 = (center_x - width/2) * img_width
                                y1 = (center_y - height/2) * img_height
                                x2 = (center_x + width/2) * img_width
                                y2 = (center_y + height/2) * img_height
                                
                                gt_boxes.append([x1, y1, x2, y2])
                                gt_classes.append(class_id)
                
                # Store results
                self.predictions.append({
                    'image': str(img_path),
                    'pred_boxes': pred_boxes,
                    'pred_classes': pred_classes,
                    'pred_scores': pred_scores,
                    'gt_boxes': gt_boxes,
                    'gt_classes': gt_classes
                })
                
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
                continue
        
        logger.info(f"Collected predictions for {len(self.predictions)} images")
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_detailed_metrics(self, iou_threshold=0.5):
        """Calculate detailed per-class metrics"""
        logger.info("Calculating detailed metrics...")
        
        # Initialize counters
        true_positives = np.zeros(self.num_classes)
        false_positives = np.zeros(self.num_classes)
        false_negatives = np.zeros(self.num_classes)
        
        all_pred_classes = []
        all_gt_classes = []
        
        for pred_data in tqdm(self.predictions, desc="Calculating metrics"):
            pred_boxes = pred_data['pred_boxes']
            pred_classes = pred_data['pred_classes']
            pred_scores = pred_data['pred_scores']
            gt_boxes = pred_data['gt_boxes']
            gt_classes = pred_data['gt_classes']
            
            # Match predictions to ground truth
            matched_gt = set()
            
            for i, (pred_box, pred_class, pred_score) in enumerate(zip(pred_boxes, pred_classes, pred_scores)):
                if pred_score < 0.5:  # Confidence threshold
                    continue
                
                best_iou = 0
                best_gt_idx = -1
                
                for j, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
                    if j in matched_gt:
                        continue
                    
                    if pred_class == gt_class:
                        iou = self.calculate_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j
                
                if best_iou >= iou_threshold and best_gt_idx != -1:
                    # True positive
                    true_positives[pred_class] += 1
                    matched_gt.add(best_gt_idx)
                    all_pred_classes.append(pred_class)
                    all_gt_classes.append(pred_class)
                else:
                    # False positive
                    false_positives[pred_class] += 1
            
            # Count false negatives (unmatched ground truth)
            for j, gt_class in enumerate(gt_classes):
                if j not in matched_gt:
                    false_negatives[gt_class] += 1
        
        # Calculate precision, recall, F1 for each class
        per_class_metrics = {}
        
        for class_id in range(self.num_classes):
            tp = true_positives[class_id]
            fp = false_positives[class_id]
            fn = false_negatives[class_id]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            
            per_class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn)
            }
        
        self.evaluation_results['detailed_metrics'] = per_class_metrics
        return per_class_metrics
    
    def analyze_pakistan_performance(self):
        """Analyze performance specifically on Pakistan-region objects"""
        logger.info("Analyzing Pakistan-specific object performance...")
        
        if 'detailed_metrics' not in self.evaluation_results:
            logger.warning("Detailed metrics not available. Run calculate_detailed_metrics first.")
            return {}
        
        detailed_metrics = self.evaluation_results['detailed_metrics']
        pakistan_metrics = {}
        
        for class_name in self.pakistan_classes:
            if class_name in detailed_metrics:
                pakistan_metrics[class_name] = detailed_metrics[class_name]
        
        # Calculate average metrics for Pakistan classes
        if pakistan_metrics:
            avg_precision = np.mean([m['precision'] for m in pakistan_metrics.values()])
            avg_recall = np.mean([m['recall'] for m in pakistan_metrics.values()])
            avg_f1 = np.mean([m['f1_score'] for m in pakistan_metrics.values()])
            
            pakistan_summary = {
                'average_precision': avg_precision,
                'average_recall': avg_recall,
                'average_f1': avg_f1,
                'classes_analyzed': len(pakistan_metrics),
                'per_class_metrics': pakistan_metrics
            }
            
            self.evaluation_results['pakistan_analysis'] = pakistan_summary
            logger.info(f"Pakistan classes analysis: Avg F1={avg_f1:.3f}, Classes={len(pakistan_metrics)}")
            
            return pakistan_summary
        
        return {}
    
    def create_confusion_matrix(self, top_n_classes=50):
        """Create confusion matrix for top N classes"""
        logger.info(f"Creating confusion matrix for top {top_n_classes} classes...")
        
        # Collect all predictions and ground truth
        all_pred_classes = []
        all_gt_classes = []
        
        for pred_data in self.predictions:
            pred_classes = pred_data['pred_classes']
            pred_scores = pred_data['pred_scores']
            gt_classes = pred_data['gt_classes']
            
            # Filter by confidence
            for pred_class, score in zip(pred_classes, pred_scores):
                if score >= 0.5:
                    all_pred_classes.append(pred_class)
            
            all_gt_classes.extend(gt_classes)
        
        # Get top N most common classes
        unique_classes, counts = np.unique(all_gt_classes, return_counts=True)
        top_class_indices = unique_classes[np.argsort(counts)[-top_n_classes:]]
        
        # Filter predictions and ground truth for top classes
        filtered_pred = []
        filtered_gt = []
        
        for pred, gt in zip(all_pred_classes, all_gt_classes):
            if gt in top_class_indices:
                filtered_pred.append(pred)
                filtered_gt.append(gt)
        
        if len(filtered_pred) == 0:
            logger.warning("No predictions found for confusion matrix")
            return
        
        # Create confusion matrix
        cm = confusion_matrix(filtered_gt, filtered_pred, labels=top_class_indices)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Get class names for labels
        class_labels = [self.class_names[i] if i < len(self.class_names) else f"class_{i}" 
                      for i in top_class_indices]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f'Confusion Matrix - Top {top_n_classes} Classes')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save confusion matrix
        cm_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved: {cm_path}")
        
        # Save confusion matrix data
        cm_data = {
            'confusion_matrix': cm.tolist(),
            'class_indices': top_class_indices.tolist(),
            'class_names': class_labels
        }
        
        cm_data_path = self.output_dir / 'confusion_matrix_data.json'
        with open(cm_data_path, 'w') as f:
            json.dump(cm_data, f, indent=2)
    
    def create_performance_visualizations(self):
        """Create comprehensive performance visualizations"""
        logger.info("Creating performance visualizations...")
        
        if 'detailed_metrics' not in self.evaluation_results:
            logger.warning("Detailed metrics not available for visualization")
            return
        
        detailed_metrics = self.evaluation_results['detailed_metrics']
        
        # Extract metrics for visualization
        class_names = list(detailed_metrics.keys())
        precisions = [detailed_metrics[cls]['precision'] for cls in class_names]
        recalls = [detailed_metrics[cls]['recall'] for cls in class_names]
        f1_scores = [detailed_metrics[cls]['f1_score'] for cls in class_names]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Precision vs Recall scatter plot
        axes[0, 0].scatter(recalls, precisions, alpha=0.6)
        axes[0, 0].set_xlabel('Recall')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_title('Precision vs Recall')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. F1 Score distribution
        axes[0, 1].hist(f1_scores, bins=20, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('F1 Score')
        axes[0, 1].set_ylabel('Number of Classes')
        axes[0, 1].set_title('F1 Score Distribution')
        axes[0, 1].axvline(np.mean(f1_scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(f1_scores):.3f}')
        axes[0, 1].legend()
        
        # 3. Top 20 classes by F1 score
        top_20_indices = np.argsort(f1_scores)[-20:]
        top_20_classes = [class_names[i] for i in top_20_indices]
        top_20_f1 = [f1_scores[i] for i in top_20_indices]
        
        axes[1, 0].barh(range(len(top_20_classes)), top_20_f1)
        axes[1, 0].set_yticks(range(len(top_20_classes)))
        axes[1, 0].set_yticklabels([cls[:20] + '...' if len(cls) > 20 else cls 
                                   for cls in top_20_classes], fontsize=8)
        axes[1, 0].set_xlabel('F1 Score')
        axes[1, 0].set_title('Top 20 Classes by F1 Score')
        
        # 4. Pakistan classes performance (if available)
        if 'pakistan_analysis' in self.evaluation_results:
            pak_metrics = self.evaluation_results['pakistan_analysis']['per_class_metrics']
            pak_classes = list(pak_metrics.keys())
            pak_f1_scores = [pak_metrics[cls]['f1_score'] for cls in pak_classes]
            
            axes[1, 1].bar(range(len(pak_classes)), pak_f1_scores, color='orange')
            axes[1, 1].set_xticks(range(len(pak_classes)))
            axes[1, 1].set_xticklabels([cls[:10] + '...' if len(cls) > 10 else cls 
                                       for cls in pak_classes], rotation=45, fontsize=8)
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].set_title('Pakistan Classes Performance')
        else:
            axes[1, 1].text(0.5, 0.5, 'Pakistan analysis\nnot available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Pakistan Classes Performance')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.output_dir / 'performance_analysis.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance visualizations saved: {viz_path}")
    
    def create_sample_predictions_visualization(self, num_samples=10):
        """Create visualization of sample predictions"""
        logger.info("Creating sample predictions visualization...")
        
        # Select random samples
        sample_indices = np.random.choice(len(self.predictions), 
                                        min(num_samples, len(self.predictions)), 
                                        replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for i, idx in enumerate(sample_indices):
            if i >= len(axes):
                break
            
            pred_data = self.predictions[idx]
            img_path = pred_data['image']
            
            try:
                # Load and display image
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Draw ground truth boxes (green)
                for box, class_id in zip(pred_data['gt_boxes'], pred_data['gt_classes']):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    cv2.putText(image, f"GT: {class_name}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw prediction boxes (red)
                for box, class_id, score in zip(pred_data['pred_boxes'], 
                                               pred_data['pred_classes'], 
                                               pred_data['pred_scores']):
                    if score < 0.5:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    cv2.putText(image, f"Pred: {class_name} ({score:.2f})", (x1, y2+15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                axes[i].imshow(image)
                axes[i].set_title(f"Sample {i+1}")
                axes[i].axis('off')
                
            except Exception as e:
                logger.warning(f"Error visualizing {img_path}: {e}")
                axes[i].text(0.5, 0.5, 'Error loading image', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"Sample {i+1} - Error")
        
        # Hide unused subplots
        for i in range(len(sample_indices), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        samples_path = self.output_dir / 'sample_predictions.png'
        plt.savefig(samples_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Sample predictions saved: {samples_path}")
    
    def save_evaluation_report(self):
        """Save comprehensive evaluation report"""
        logger.info("Saving evaluation report...")
        
        # Create comprehensive report
        report = {
            'model_path': str(self.model_path),
            'data_config': str(self.data_yaml_path),
            'evaluation_date': datetime.now().isoformat(),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'results': self.evaluation_results
        }
        
        # Save JSON report
        report_path = self.output_dir / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create human-readable summary
        summary_path = self.output_dir / 'evaluation_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("YOLOv8 Model Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model: {self.model_path.name}\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Classes: {self.num_classes}\n\n")
            
            # Official metrics
            if 'official_metrics' in self.evaluation_results:
                metrics = self.evaluation_results['official_metrics']
                f.write("Official Metrics:\n")
                f.write("-" * 20 + "\n")
                f.write(f"mAP@0.5: {metrics.get('mAP50', 0):.4f}\n")
                f.write(f"mAP@0.5:0.95: {metrics.get('mAP50_95', 0):.4f}\n")
                f.write(f"Precision: {metrics.get('precision', 0):.4f}\n")
                f.write(f"Recall: {metrics.get('recall', 0):.4f}\n\n")
            
            # Pakistan analysis
            if 'pakistan_analysis' in self.evaluation_results:
                pak_analysis = self.evaluation_results['pakistan_analysis']
                f.write("Pakistan-Specific Analysis:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Classes Analyzed: {pak_analysis.get('classes_analyzed', 0)}\n")
                f.write(f"Average Precision: {pak_analysis.get('average_precision', 0):.4f}\n")
                f.write(f"Average Recall: {pak_analysis.get('average_recall', 0):.4f}\n")
                f.write(f"Average F1 Score: {pak_analysis.get('average_f1', 0):.4f}\n\n")
            
            f.write("Generated Files:\n")
            f.write("-" * 15 + "\n")
            f.write("- evaluation_report.json (detailed results)\n")
            f.write("- confusion_matrix.png (confusion matrix)\n")
            f.write("- performance_analysis.png (performance charts)\n")
            f.write("- sample_predictions.png (sample visualizations)\n")
        
        logger.info(f"Evaluation report saved: {report_path}")
        logger.info(f"Evaluation summary saved: {summary_path}")
    
    def run_complete_evaluation(self, test_images_dir=None, test_labels_dir=None):
        """Run complete evaluation pipeline"""
        logger.info("Starting complete model evaluation...")
        
        # 1. Official evaluation
        self.run_official_evaluation()
        
        # 2. Detailed analysis (if test data provided)
        if test_images_dir and test_labels_dir:
            self.collect_predictions_and_ground_truth(test_images_dir, test_labels_dir)
            self.calculate_detailed_metrics()
            self.analyze_pakistan_performance()
            self.create_confusion_matrix()
            self.create_sample_predictions_visualization()
        
        # 3. Create visualizations
        self.create_performance_visualizations()
        
        # 4. Save comprehensive report
        self.save_evaluation_report()
        
        logger.info("Complete evaluation finished!")
        logger.info(f"Results saved to: {self.output_dir}")
        
        return self.evaluation_results

def main():
    """Main function to run model evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate YOLOv8 model')
    parser.add_argument('--model', type=str, required=True, help='Path to model file (.pt)')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--test-images', type=str, help='Path to test images directory')
    parser.add_argument('--test-labels', type=str, help='Path to test labels directory')
    parser.add_argument('--output', type=str, default='evaluation_results', help='Output directory')
    
    args = parser.parse_args()
    
    print("🔍 Starting YOLOv8 Model Evaluation")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model, args.data, args.output)
    
    print(f"📊 Model: {args.model}")
    print(f"📁 Data: {args.data}")
    print(f"🏷️  Classes: {evaluator.num_classes}")
    
    try:
        # Run evaluation
        results = evaluator.run_complete_evaluation(args.test_images, args.test_labels)
        
        print("\n✅ Evaluation completed successfully!")
        
        # Print summary
        if 'official_metrics' in results:
            metrics = results['official_metrics']
            print(f"\n📈 Key Metrics:")
            print(f"   mAP@0.5: {metrics.get('mAP50', 0):.4f}")
            print(f"   mAP@0.5:0.95: {metrics.get('mAP50_95', 0):.4f}")
            print(f"   Precision: {metrics.get('precision', 0):.4f}")
            print(f"   Recall: {metrics.get('recall', 0):.4f}")
        
        if 'pakistan_analysis' in results:
            pak_metrics = results['pakistan_analysis']
            print(f"\n🇵🇰 Pakistan Classes:")
            print(f"   Analyzed: {pak_metrics.get('classes_analyzed', 0)} classes")
            print(f"   Avg F1: {pak_metrics.get('average_f1', 0):.4f}")
        
        print(f"\n📁 Results saved to: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
