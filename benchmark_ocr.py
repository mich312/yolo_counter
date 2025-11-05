#!/usr/bin/env python3
"""
OCR Performance Benchmark

Compares performance of PaddleOCR vs EasyOCR for timestamp extraction
"""

import time
import numpy as np
import cv2
import sys
from typing import Optional, Dict, List
import traceback

class OCRBenchmark:
    """Benchmark OCR performance"""

    def __init__(self):
        self.results = {}

    def create_test_images(self) -> List[np.ndarray]:
        """Create synthetic test images with timestamps"""
        images = []

        # Test cases with different timestamp formats
        test_texts = [
            "05.11.2025 14:30",
            "31.12.2024 23:59",
            "01.01.2025 00:00",
            "15.06.2025 12:45",
            "28.02.2025 18:20",
        ]

        for text in test_texts:
            # Create white background
            img = np.ones((120, 500, 3), dtype=np.uint8) * 255

            # Add text with OpenCV
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            font_thickness = 2

            # Calculate text size for centering
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, font_thickness
            )

            # Center text
            x = (img.shape[1] - text_width) // 2
            y = (img.shape[0] + text_height) // 2

            cv2.putText(img, text, (x, y), font, font_scale,
                       (0, 0, 0), font_thickness, cv2.LINE_AA)

            images.append(img)

        return images

    def benchmark_paddleocr(self, images: List[np.ndarray]) -> Dict:
        """Benchmark PaddleOCR performance"""
        print("\n" + "="*60)
        print("BENCHMARKING PADDLEOCR")
        print("="*60)

        results = {
            'name': 'PaddleOCR',
            'available': False,
            'init_time': 0,
            'avg_time': 0,
            'min_time': 0,
            'max_time': 0,
            'total_time': 0,
            'success_rate': 0,
            'error': None
        }

        try:
            from src.ocr_paddle import PaddleTimestampExtractor

            # Benchmark initialization
            print("\n1. Initialization Time")
            init_start = time.time()
            extractor = PaddleTimestampExtractor(['en'])
            extractor.load_reader()
            init_time = time.time() - init_start
            results['init_time'] = init_time
            results['available'] = True
            print(f"   Initialization: {init_time:.3f}s")

            # Benchmark extraction
            print("\n2. Timestamp Extraction Performance")
            extraction_times = []
            successful_extractions = 0

            for i, img in enumerate(images):
                start = time.time()
                timestamp = extractor.extract_timestamp(img)
                elapsed = time.time() - start
                extraction_times.append(elapsed)

                if timestamp is not None:
                    successful_extractions += 1
                    print(f"   Image {i+1}: {elapsed*1000:.1f}ms ✓ {timestamp}")
                else:
                    print(f"   Image {i+1}: {elapsed*1000:.1f}ms ✗ (no timestamp)")

            # Calculate statistics
            results['avg_time'] = np.mean(extraction_times)
            results['min_time'] = np.min(extraction_times)
            results['max_time'] = np.max(extraction_times)
            results['total_time'] = np.sum(extraction_times)
            results['success_rate'] = (successful_extractions / len(images)) * 100
            results['extraction_times'] = extraction_times

            print(f"\n3. Statistics")
            print(f"   Average: {results['avg_time']*1000:.1f}ms")
            print(f"   Min: {results['min_time']*1000:.1f}ms")
            print(f"   Max: {results['max_time']*1000:.1f}ms")
            print(f"   Total: {results['total_time']:.3f}s")
            print(f"   Success Rate: {results['success_rate']:.1f}%")

        except ImportError as e:
            results['error'] = f"Not installed: {e}"
            print(f"⚠ PaddleOCR not available: {e}")
        except Exception as e:
            results['error'] = str(e)
            print(f"✗ PaddleOCR benchmark failed: {e}")
            traceback.print_exc()

        return results

    def benchmark_easyocr(self, images: List[np.ndarray]) -> Dict:
        """Benchmark EasyOCR performance"""
        print("\n" + "="*60)
        print("BENCHMARKING EASYOCR")
        print("="*60)

        results = {
            'name': 'EasyOCR',
            'available': False,
            'init_time': 0,
            'avg_time': 0,
            'min_time': 0,
            'max_time': 0,
            'total_time': 0,
            'success_rate': 0,
            'error': None
        }

        try:
            from src.ocr import TimestampExtractor

            # Benchmark initialization
            print("\n1. Initialization Time")
            init_start = time.time()
            extractor = TimestampExtractor(['en'])
            extractor.load_reader()
            init_time = time.time() - init_start
            results['init_time'] = init_time
            results['available'] = True
            print(f"   Initialization: {init_time:.3f}s")

            # Benchmark extraction
            print("\n2. Timestamp Extraction Performance")
            extraction_times = []
            successful_extractions = 0

            for i, img in enumerate(images):
                start = time.time()
                timestamp = extractor.extract_timestamp(img)
                elapsed = time.time() - start
                extraction_times.append(elapsed)

                if timestamp is not None:
                    successful_extractions += 1
                    print(f"   Image {i+1}: {elapsed*1000:.1f}ms ✓ {timestamp}")
                else:
                    print(f"   Image {i+1}: {elapsed*1000:.1f}ms ✗ (no timestamp)")

            # Calculate statistics
            results['avg_time'] = np.mean(extraction_times)
            results['min_time'] = np.min(extraction_times)
            results['max_time'] = np.max(extraction_times)
            results['total_time'] = np.sum(extraction_times)
            results['success_rate'] = (successful_extractions / len(images)) * 100
            results['extraction_times'] = extraction_times

            print(f"\n3. Statistics")
            print(f"   Average: {results['avg_time']*1000:.1f}ms")
            print(f"   Min: {results['min_time']*1000:.1f}ms")
            print(f"   Max: {results['max_time']*1000:.1f}ms")
            print(f"   Total: {results['total_time']:.3f}s")
            print(f"   Success Rate: {results['success_rate']:.1f}%")

        except ImportError as e:
            results['error'] = f"Not installed: {e}"
            print(f"⚠ EasyOCR not available: {e}")
        except Exception as e:
            results['error'] = str(e)
            print(f"✗ EasyOCR benchmark failed: {e}")
            traceback.print_exc()

        return results

    def generate_comparison_report(self, paddle_results: Dict, easy_results: Dict):
        """Generate comparison report"""
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON REPORT")
        print("="*60)

        # Table header
        print(f"\n{'Metric':<25} {'PaddleOCR':>15} {'EasyOCR':>15} {'Winner':>12}")
        print("-" * 70)

        # Availability
        paddle_avail = "✓ Available" if paddle_results['available'] else "✗ N/A"
        easy_avail = "✓ Available" if easy_results['available'] else "✗ N/A"
        print(f"{'Availability':<25} {paddle_avail:>15} {easy_avail:>15} {'-':>12}")

        if paddle_results['available'] and easy_results['available']:
            # Initialization time
            paddle_init = f"{paddle_results['init_time']:.2f}s"
            easy_init = f"{easy_results['init_time']:.2f}s"
            init_winner = "PaddleOCR" if paddle_results['init_time'] < easy_results['init_time'] else "EasyOCR"
            print(f"{'Initialization Time':<25} {paddle_init:>15} {easy_init:>15} {init_winner:>12}")

            # Average extraction time
            paddle_avg = f"{paddle_results['avg_time']*1000:.1f}ms"
            easy_avg = f"{easy_results['avg_time']*1000:.1f}ms"
            avg_winner = "PaddleOCR" if paddle_results['avg_time'] < easy_results['avg_time'] else "EasyOCR"
            print(f"{'Avg Extraction Time':<25} {paddle_avg:>15} {easy_avg:>15} {avg_winner:>12}")

            # Min extraction time
            paddle_min = f"{paddle_results['min_time']*1000:.1f}ms"
            easy_min = f"{easy_results['min_time']*1000:.1f}ms"
            min_winner = "PaddleOCR" if paddle_results['min_time'] < easy_results['min_time'] else "EasyOCR"
            print(f"{'Min Extraction Time':<25} {paddle_min:>15} {easy_min:>15} {min_winner:>12}")

            # Max extraction time
            paddle_max = f"{paddle_results['max_time']*1000:.1f}ms"
            easy_max = f"{easy_results['max_time']*1000:.1f}ms"
            max_winner = "PaddleOCR" if paddle_results['max_time'] < easy_results['max_time'] else "EasyOCR"
            print(f"{'Max Extraction Time':<25} {paddle_max:>15} {easy_max:>15} {max_winner:>12}")

            # Success rate
            paddle_success = f"{paddle_results['success_rate']:.1f}%"
            easy_success = f"{easy_results['success_rate']:.1f}%"
            success_winner = "PaddleOCR" if paddle_results['success_rate'] > easy_results['success_rate'] else "EasyOCR"
            print(f"{'Success Rate':<25} {paddle_success:>15} {easy_success:>15} {success_winner:>12}")

            # Speedup calculation
            print("\n" + "-" * 70)
            speedup = easy_results['avg_time'] / paddle_results['avg_time']
            print(f"\n🚀 Speedup: PaddleOCR is {speedup:.2f}x faster than EasyOCR")

            init_speedup = easy_results['init_time'] / paddle_results['init_time']
            print(f"⚡ Initialization: PaddleOCR is {init_speedup:.2f}x faster")

        elif paddle_results['available']:
            print(f"\n⚠ Only PaddleOCR available for testing")
            print(f"  Install EasyOCR: pip install easyocr")
        elif easy_results['available']:
            print(f"\n⚠ Only EasyOCR available for testing")
            print(f"  Install PaddleOCR: pip install 'paddleocr>=2.8,<3.0'")
        else:
            print(f"\n✗ Neither OCR engine available")
            print(f"  PaddleOCR: {paddle_results.get('error', 'Unknown error')}")
            print(f"  EasyOCR: {easy_results.get('error', 'Unknown error')}")

        # Errors
        if paddle_results.get('error'):
            print(f"\nPaddleOCR Error: {paddle_results['error']}")
        if easy_results.get('error'):
            print(f"\nEasyOCR Error: {easy_results['error']}")

    def run(self):
        """Run complete benchmark"""
        print("="*60)
        print("OCR PERFORMANCE BENCHMARK")
        print("="*60)

        # Create test images
        print("\nCreating test images...")
        images = self.create_test_images()
        print(f"✓ Created {len(images)} test images")

        # Save sample image for reference
        cv2.imwrite('/tmp/sample_timestamp.png', images[0])
        print("✓ Saved sample image to /tmp/sample_timestamp.png")

        # Benchmark both engines
        paddle_results = self.benchmark_paddleocr(images)
        easy_results = self.benchmark_easyocr(images)

        # Generate comparison report
        self.generate_comparison_report(paddle_results, easy_results)

        # Store results
        self.results = {
            'paddleocr': paddle_results,
            'easyocr': easy_results,
            'test_count': len(images)
        }

        return self.results


def main():
    """Main entry point"""
    benchmark = OCRBenchmark()
    results = benchmark.run()

    # Determine exit code
    if results['paddleocr']['available'] or results['easyocr']['available']:
        print("\n✓ Benchmark completed successfully")
        return 0
    else:
        print("\n✗ Benchmark failed - no OCR engines available")
        return 1


if __name__ == '__main__':
    sys.exit(main())
