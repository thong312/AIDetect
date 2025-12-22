"""
Performance monitoring and logging for ensemble models
Track accuracy, latency, memory usage, and other metrics
"""

import time
import psutil
import logging
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path


class ModelMetrics:
    """Track individual metrics"""
    def __init__(self, name: str):
        self.name = name
        self.predictions = 0
        self.correct = 0
        self.total_inference_time = 0
        self.inference_times = []
        self.confidences = []
        self.memory_usage = []

    def add_prediction(self, correct: bool, inference_time: float, confidence: float, memory: float):
        """Record a prediction"""
        self.predictions += 1
        if correct:
            self.correct += 1
        self.total_inference_time += inference_time
        self.inference_times.append(inference_time)
        self.confidences.append(confidence)
        self.memory_usage.append(memory)

    def get_accuracy(self) -> float:
        """Get accuracy percentage"""
        if self.predictions == 0:
            return 0
        return (self.correct / self.predictions) * 100

    def get_avg_inference_time(self) -> float:
        """Get average inference time in ms"""
        if not self.inference_times:
            return 0
        return (sum(self.inference_times) / len(self.inference_times)) * 1000

    def get_avg_confidence(self) -> float:
        """Get average confidence"""
        if not self.confidences:
            return 0
        return sum(self.confidences) / len(self.confidences)

    def get_summary(self) -> Dict:
        """Get metrics summary"""
        return {
            'name': self.name,
            'predictions': self.predictions,
            'accuracy': round(self.get_accuracy(), 2),
            'avg_inference_time_ms': round(self.get_avg_inference_time(), 2),
            'avg_confidence': round(self.get_avg_confidence(), 2),
            'total_inference_time_s': round(self.total_inference_time, 2),
            'avg_memory_mb': round(sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0, 2)
        }


class EnsembleMonitor:
    """
    Monitor ensemble model performance
    Tracks accuracy, latency, memory, and confidence
    """
    def __init__(self, log_file: Optional[str] = None, enable_detailed_logging: bool = True):
        self.enable_detailed_logging = enable_detailed_logging

        # Setup logging
        self.logger = logging.getLogger('EnsembleMonitor')
        self.logger.setLevel(logging.INFO)

        # File handler
        if log_file is None:
            log_file = f"logs/ensemble_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        Path(log_file).parent.mkdir(exist_ok=True)

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # Metrics
        self.vit_metrics = ModelMetrics("ViT-B/16")
        self.transunet_metrics = ModelMetrics("TransUNet")
        self.ensemble_metrics = ModelMetrics("Ensemble")

        # Session info
        self.start_time = datetime.now()
        self.predictions_count = 0

    def record_prediction(
        self,
        ensemble_correct: bool,
        vit_correct: bool,
        transunet_correct: bool,
        ensemble_confidence: float,
        vit_confidence: float,
        transunet_confidence: float,
        inference_time: float,
        image_name: str = ""
    ):
        """Record a prediction and its metrics"""
        # Get current memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        # Record metrics
        self.vit_metrics.add_prediction(
            vit_correct, inference_time, vit_confidence, memory_mb
        )
        self.transunet_metrics.add_prediction(
            transunet_correct, inference_time, transunet_confidence, memory_mb
        )
        self.ensemble_metrics.add_prediction(
            ensemble_correct, inference_time, ensemble_confidence, memory_mb
        )

        self.predictions_count += 1

        # Detailed logging
        if self.enable_detailed_logging:
            self.logger.info(
                f"Prediction #{self.predictions_count} | Image: {image_name} | "
                f"Ensemble: {'✓' if ensemble_correct else '✗'} ({ensemble_confidence:.1f}%) | "
                f"ViT: {'✓' if vit_correct else '✗'} ({vit_confidence:.1f}%) | "
                f"TransUNet: {'✓' if transunet_correct else '✗'} ({transunet_confidence:.1f}%) | "
                f"Time: {inference_time*1000:.2f}ms | Memory: {memory_mb:.1f}MB"
            )

    def print_summary(self):
        """Print detailed summary"""
        print("\n" + "="*80)
        print("ENSEMBLE MONITOR SUMMARY")
        print("="*80)

        elapsed = (datetime.now() - self.start_time).total_seconds()

        print(f"\nSession Duration: {elapsed:.2f}s")
        print(f"Total Predictions: {self.predictions_count}")
        print(f"Average Throughput: {self.predictions_count/elapsed if elapsed > 0 else 0:.2f} predictions/sec")

        print("\n" + "-"*80)
        print("MODEL PERFORMANCE")
        print("-"*80)

        for metrics in [self.ensemble_metrics, self.vit_metrics, self.transunet_metrics]:
            summary = metrics.get_summary()
            print(f"\n{summary['name']}:")
            print(f"  Accuracy:              {summary['accuracy']:.2f}%")
            print(f"  Predictions:           {summary['predictions']}")
            print(f"  Avg Inference Time:    {summary['avg_inference_time_ms']:.2f}ms")
            print(f"  Avg Confidence:        {summary['avg_confidence']:.2f}%")
            print(f"  Total Inference Time:  {summary['total_inference_time_s']:.2f}s")
            print(f"  Avg Memory Usage:      {summary['avg_memory_mb']:.2f}MB")

        # Ensemble improvement
        vit_acc = self.vit_metrics.get_accuracy()
        transunet_acc = self.transunet_metrics.get_accuracy()
        ensemble_acc = self.ensemble_metrics.get_accuracy()

        print("\n" + "-"*80)
        print("ENSEMBLE IMPROVEMENT")
        print("-"*80)
        print(f"ViT Accuracy:          {vit_acc:.2f}%")
        print(f"TransUNet Accuracy:    {transunet_acc:.2f}%")
        print(f"Ensemble Accuracy:     {ensemble_acc:.2f}%")

        if vit_acc > 0:
            improvement = ((ensemble_acc - vit_acc) / vit_acc) * 100
            print(f"Improvement vs ViT:    {improvement:+.2f}%")

        if transunet_acc > 0:
            improvement = ((ensemble_acc - transunet_acc) / transunet_acc) * 100
            print(f"Improvement vs TransUNet: {improvement:+.2f}%")

        print("\n" + "="*80 + "\n")

    def export_report(self, output_file: str = "ensemble_report.json"):
        """Export detailed report to JSON"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'session_duration_s': (datetime.now() - self.start_time).total_seconds(),
            'total_predictions': self.predictions_count,
            'vit_metrics': self.vit_metrics.get_summary(),
            'transunet_metrics': self.transunet_metrics.get_summary(),
            'ensemble_metrics': self.ensemble_metrics.get_summary(),
            'system_info': {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent
            }
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Report exported to {output_file}")
        return report


class PerformanceOptimizer:
    """Analyze and optimize ensemble performance"""
    def __init__(self, monitor: EnsembleMonitor):
        self.monitor = monitor

    def get_bottleneck(self) -> str:
        """Identify performance bottleneck"""
        vit_time = self.monitor.vit_metrics.get_avg_inference_time()
        transunet_time = self.monitor.transunet_metrics.get_avg_inference_time()

        if vit_time > transunet_time:
            return f"ViT ({vit_time:.2f}ms) is slower"
        else:
            return f"TransUNet ({transunet_time:.2f}ms) is slower"

    def get_recommendations(self) -> List[str]:
        """Get optimization recommendations"""
        recommendations = []

        vit_acc = self.monitor.vit_metrics.get_accuracy()
        transunet_acc = self.monitor.transunet_metrics.get_accuracy()
        ensemble_acc = self.monitor.ensemble_metrics.get_accuracy()

        # Accuracy recommendations
        if vit_acc > transunet_acc + 10:
            recommendations.append(
                f"ViT is significantly better ({vit_acc:.1f}% vs {transunet_acc:.1f}%). "
                "Consider increasing ViT weight."
            )
        elif transunet_acc > vit_acc + 10:
            recommendations.append(
                f"TransUNet is significantly better ({transunet_acc:.1f}% vs {vit_acc:.1f}%). "
                "Consider increasing TransUNet weight."
            )

        # Latency recommendations
        vit_time = self.monitor.vit_metrics.get_avg_inference_time()
        transunet_time = self.monitor.transunet_metrics.get_avg_inference_time()
        ensemble_time = self.monitor.ensemble_metrics.get_avg_inference_time()

        if ensemble_time > vit_time + transunet_time:
            recommendations.append(
                "Ensemble overhead detected. Consider batch processing or model quantization."
            )

        if transunet_time > vit_time * 3:
            recommendations.append(
                "TransUNet is much slower. Consider using ViT-only or model distillation."
            )

        # Confidence recommendations
        vit_conf = self.monitor.vit_metrics.get_avg_confidence()
        transunet_conf = self.monitor.transunet_metrics.get_avg_confidence()

        if vit_conf < 60 or transunet_conf < 60:
            recommendations.append(
                "Low confidence scores detected. Consider calibration or data augmentation."
            )

        return recommendations if recommendations else ["No optimization needed!"]


if __name__ == "__main__":
    # Example usage
    monitor = EnsembleMonitor(enable_detailed_logging=True)
    optimizer = PerformanceOptimizer(monitor)

    # Simulate some predictions
    for i in range(10):
        monitor.record_prediction(
            ensemble_correct=True,
            vit_correct=True,
            transunet_correct=False,
            ensemble_confidence=92.5,
            vit_confidence=91.2,
            transunet_confidence=88.7,
            inference_time=0.05,
            image_name=f"test_{i}.jpg"
        )

    # Print results
    monitor.print_summary()

    print("\nOptimization Recommendations:")
    for i, rec in enumerate(optimizer.get_recommendations(), 1):
        print(f"{i}. {rec}")

    print(f"\nBottleneck: {optimizer.get_bottleneck()}")

    # Export report
    monitor.export_report("ensemble_report.json")
