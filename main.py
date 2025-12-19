#!/usr/bin/env python3
"""
Main entry point for the Enhanced U-Net Skin Lesion Segmentation Project.

This script provides a clean interface to run different components of the project:
- Original U-Net training
- Enhanced Attention U-Net training
- Inference on new images
- Model evaluation
"""

import os
import sys
import argparse
from datetime import datetime

def print_header():
    """Print project header information."""
    print("=" * 80)
    print("U-Net Skin Lesion Segmentation - Enhanced Version")
    print("Advanced Medical Image Analysis with Attention Mechanisms")
    print("=" * 80)
    print()

def main():
    print_header()

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Enhanced U-Net Skin Lesion Segmentation')

    parser.add_argument('--mode', type=str, default='enhanced',
                       choices=['original', 'enhanced', 'inference', 'evaluate'],
                       help='Mode to run: original (basic U-Net), enhanced (Attention U-Net), inference, or evaluate')

    parser.add_argument('--data_dir', type=str, default='data/isic_2016',
                       help='Path to ISIC 2016 dataset directory')

    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model for inference/evaluation')

    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to image for inference')

    parser.add_argument('--results_dir', type=str, default='Results_Enhanced',
                       help='Directory to save results')

    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')

    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate for training')

    parser.add_argument('--folds', type=int, default=5,
                       help='Number of folds for cross-validation')

    parser.add_argument('--no_augmentation', action='store_true',
                       help='Disable data augmentation')

    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose output')

    args = parser.parse_args()

    # Add project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)

    print(f"Project Root: {project_root}")
    print(f"Mode: {args.mode}")
    print(f"Results Directory: {args.results_dir}")
    print()

    # Create results directory if it doesn't exist
    results_dir = os.path.join(project_root, args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Create models directory
    models_dir = os.path.join(project_root, 'models_enhanced')
    os.makedirs(models_dir, exist_ok=True)

    # Run the appropriate mode
    if args.mode == 'original':
        print("Running original U-Net training...")
        from src.train import main as original_train
        original_train()

    elif args.mode == 'enhanced':
        print("Running enhanced Attention U-Net training...")
        from src.train_enhanced import main as enhanced_train

        # Set up configuration for enhanced training
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

        if args.debug:
            print("Debug mode enabled - verbose output")
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

        enhanced_train()

    elif args.mode == 'inference':
        print("Running inference mode...")
        if not args.model_path or not args.image_path:
            print("Error: Both --model_path and --image_path are required for inference mode.")
            return

        from src.utils.inference import run_inference
        run_inference(args.model_path, args.image_path, results_dir)

    elif args.mode == 'evaluate':
        print("Running evaluation mode...")
        if not args.model_path:
            print("Error: --model_path is required for evaluation mode.")
            return

        from src.evaluation.evaluate import run_evaluation
        run_evaluation(args.model_path, args.data_dir, results_dir)

    else:
        print(f"Unknown mode: {args.mode}")

    print(f"\nAll operations completed. Results saved to: {results_dir}")

def show_usage():
    """Show usage examples."""
    print("\nUsage Examples:")
    print("  python main.py --mode enhanced                    # Run enhanced training")
    print("  python main.py --mode original                    # Run original training")
    print("  python main.py --mode inference --model_path models/best_model.keras --image_path test.jpg")
    print("  python main.py --mode evaluate --model_path models/best_model.keras")
    print("  python main.py --mode enhanced --epochs 50 --batch_size 8 --learning_rate 1e-3")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        show_usage()
    finally:
        print("\nThank you for using the Enhanced U-Net Skin Lesion Segmentation system!")
