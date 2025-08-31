import pandas as pd
import numpy as np
import ast
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_data():
    """Load both original and interpolated data for comparison"""
    try:
        original_data = pd.read_csv('test.csv')
        interpolated_data = pd.read_csv('test_interpolated.csv')
        return original_data, interpolated_data
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

def calculate_detection_metrics(original_data, interpolated_data):
    """Calculate detection performance metrics"""
    metrics = {}
    
    # Basic statistics
    total_frames_original = len(original_data)
    total_frames_interpolated = len(interpolated_data)
    unique_vehicles_original = len(original_data['car_id'].unique())
    unique_vehicles_interpolated = len(interpolated_data['car_id'].unique())
    
    # Frame coverage
    frame_range_original = original_data['frame_nmr'].max() - original_data['frame_nmr'].min() + 1
    frame_range_interpolated = interpolated_data['frame_nmr'].max() - interpolated_data['frame_nmr'].min() + 1
    
    # Detection rates
    detection_rate_original = (total_frames_original / frame_range_original) * 100 if frame_range_original > 0 else 0
    detection_rate_interpolated = (total_frames_interpolated / frame_range_interpolated) * 100 if frame_range_interpolated > 0 else 0
    
    metrics['detection'] = {
        'total_detections_original': total_frames_original,
        'total_detections_interpolated': total_frames_interpolated,
        'unique_vehicles': unique_vehicles_original,
        'frame_range': frame_range_original,
        'detection_rate_original': detection_rate_original,
        'detection_rate_interpolated': detection_rate_interpolated,
        'interpolation_improvement': detection_rate_interpolated - detection_rate_original
    }
    
    return metrics

def calculate_recognition_metrics(data):
    """Calculate OCR and recognition performance metrics"""
    metrics = {}
    
    # Filter out rows with valid license numbers (not '0' or empty)
    valid_recognitions = data[
        (data['license_number'] != '0') & 
        (data['license_number'].notna()) & 
        (data['license_number'] != '')
    ]
    
    total_detections = len(data)
    successful_recognitions = len(valid_recognitions)
    
    # OCR success rate
    ocr_success_rate = (successful_recognitions / total_detections) * 100 if total_detections > 0 else 0
    
    # Confidence score analysis
    if 'license_plate_bbox_score' in data.columns:
        bbox_scores = pd.to_numeric(data['license_plate_bbox_score'], errors='coerce').dropna()
        avg_bbox_confidence = bbox_scores.mean()
        median_bbox_confidence = bbox_scores.median()
    else:
        avg_bbox_confidence = median_bbox_confidence = 0
    
    if 'license_number_score' in data.columns:
        text_scores = pd.to_numeric(data['license_number_score'], errors='coerce').dropna()
        text_scores = text_scores[text_scores > 0]  # Exclude interpolated scores (0)
        avg_text_confidence = text_scores.mean()
        median_text_confidence = text_scores.median()
    else:
        avg_text_confidence = median_text_confidence = 0
    
    # License plate format analysis
    license_patterns = []
    if successful_recognitions > 0:
        for license in valid_recognitions['license_number']:
            # Analyze license plate patterns
            if re.match(r'^[A-Z]{2,3}\d{2,4}[A-Z]?$', str(license)):
                license_patterns.append('Standard')
            elif re.match(r'^\d{2,4}[A-Z]{2,3}$', str(license)):
                license_patterns.append('Numeric-Alpha')
            else:
                license_patterns.append('Other')
    
    pattern_distribution = Counter(license_patterns) if license_patterns else {}
    
    metrics['recognition'] = {
        'total_detections': total_detections,
        'successful_recognitions': successful_recognitions,
        'ocr_success_rate': ocr_success_rate,
        'avg_bbox_confidence': avg_bbox_confidence,
        'median_bbox_confidence': median_bbox_confidence,
        'avg_text_confidence': avg_text_confidence,
        'median_text_confidence': median_text_confidence,
        'license_pattern_distribution': dict(pattern_distribution)
    }
    
    return metrics

def calculate_tracking_metrics(data):
    """Calculate vehicle tracking performance metrics"""
    metrics = {}
    
    # Per-vehicle tracking analysis
    vehicle_stats = {}
    for vehicle_id in data['car_id'].unique():
        vehicle_data = data[data['car_id'] == vehicle_id]
        
        frames = sorted(vehicle_data['frame_nmr'].tolist())
        tracking_duration = max(frames) - min(frames) + 1
        detection_count = len(frames)
        
        # Calculate gaps in tracking
        gaps = []
        for i in range(1, len(frames)):
            gap = frames[i] - frames[i-1] - 1
            if gap > 0:
                gaps.append(gap)
        
        # Recognition success for this vehicle
        valid_recognitions = vehicle_data[
            (vehicle_data['license_number'] != '0') & 
            (vehicle_data['license_number'].notna()) & 
            (vehicle_data['license_number'] != '')
        ]
        
        vehicle_stats[vehicle_id] = {
            'tracking_duration': tracking_duration,
            'detection_count': detection_count,
            'tracking_consistency': (detection_count / tracking_duration) * 100 if tracking_duration > 0 else 0,
            'total_gaps': len(gaps),
            'avg_gap_size': np.mean(gaps) if gaps else 0,
            'max_gap_size': max(gaps) if gaps else 0,
            'recognition_count': len(valid_recognitions),
            'recognition_rate': (len(valid_recognitions) / len(vehicle_data)) * 100 if len(vehicle_data) > 0 else 0
        }
    
    # Overall tracking statistics
    tracking_durations = [stats['tracking_duration'] for stats in vehicle_stats.values()]
    tracking_consistencies = [stats['tracking_consistency'] for stats in vehicle_stats.values()]
    recognition_rates = [stats['recognition_rate'] for stats in vehicle_stats.values()]
    
    metrics['tracking'] = {
        'total_vehicles': len(vehicle_stats),
        'avg_tracking_duration': np.mean(tracking_durations) if tracking_durations else 0,
        'median_tracking_duration': np.median(tracking_durations) if tracking_durations else 0,
        'avg_tracking_consistency': np.mean(tracking_consistencies) if tracking_consistencies else 0,
        'avg_recognition_rate_per_vehicle': np.mean(recognition_rates) if recognition_rates else 0,
        'vehicle_details': vehicle_stats
    }
    
    return metrics

def calculate_interpolation_impact(original_data, interpolated_data):
    """Analyze the impact of data interpolation"""
    metrics = {}
    
    original_frames = set(original_data['frame_nmr'])
    interpolated_frames = set(interpolated_data['frame_nmr'])
    
    # Frames added by interpolation
    added_frames = interpolated_frames - original_frames
    
    # Data augmentation statistics
    original_count = len(original_data)
    interpolated_count = len(interpolated_data)
    augmentation_ratio = (interpolated_count - original_count) / original_count * 100 if original_count > 0 else 0
    
    metrics['interpolation'] = {
        'original_detection_count': original_count,
        'interpolated_detection_count': interpolated_count,
        'frames_added': len(added_frames),
        'augmentation_ratio': augmentation_ratio,
        'data_completeness_improvement': len(added_frames) / len(interpolated_frames) * 100 if interpolated_frames else 0
    }
    
    return metrics

def generate_performance_report(metrics):
    """Generate a comprehensive performance report"""
    print("=" * 80)
    print("LICENSE PLATE DETECTION & RECOGNITION SYSTEM - PERFORMANCE REPORT")
    print("=" * 80)
    print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Detection Performance
    print("🎯 DETECTION PERFORMANCE")
    print("-" * 40)
    det = metrics['detection']
    print(f"Original Detections: {det['total_detections_original']:,}")
    print(f"After Interpolation: {det['total_detections_interpolated']:,}")
    print(f"Unique Vehicles Tracked: {det['unique_vehicles']}")
    print(f"Frame Range Covered: {det['frame_range']:,} frames")
    print(f"Detection Rate (Original): {det['detection_rate_original']:.2f}%")
    print(f"Detection Rate (Interpolated): {det['detection_rate_interpolated']:.2f}%")
    print(f"Improvement from Interpolation: +{det['interpolation_improvement']:.2f}%")
    print()
    
    # Recognition Performance
    print("🔍 RECOGNITION PERFORMANCE")
    print("-" * 40)
    rec = metrics['recognition']
    print(f"Total License Plate Detections: {rec['total_detections']:,}")
    print(f"Successful Text Recognition: {rec['successful_recognitions']:,}")
    print(f"OCR Success Rate: {rec['ocr_success_rate']:.2f}%")
    print(f"Average Bounding Box Confidence: {rec['avg_bbox_confidence']:.3f}")
    print(f"Average Text Recognition Confidence: {rec['avg_text_confidence']:.3f}")
    print("License Plate Pattern Distribution:")
    for pattern, count in rec['license_pattern_distribution'].items():
        print(f"  {pattern}: {count} ({count/rec['successful_recognitions']*100:.1f}%)")
    print()
    
    # Tracking Performance
    print("📍 TRACKING PERFORMANCE")
    print("-" * 40)
    track = metrics['tracking']
    print(f"Total Vehicles Tracked: {track['total_vehicles']}")
    print(f"Average Tracking Duration: {track['avg_tracking_duration']:.1f} frames")
    print(f"Median Tracking Duration: {track['median_tracking_duration']:.1f} frames")
    print(f"Average Tracking Consistency: {track['avg_tracking_consistency']:.2f}%")
    print(f"Average Recognition Rate per Vehicle: {track['avg_recognition_rate_per_vehicle']:.2f}%")
    print()
    
    # Interpolation Impact
    print("📈 INTERPOLATION IMPACT")
    print("-" * 40)
    interp = metrics['interpolation']
    print(f"Original Detection Count: {interp['original_detection_count']:,}")
    print(f"After Interpolation: {interp['interpolated_detection_count']:,}")
    print(f"Frames Added by Interpolation: {interp['frames_added']:,}")
    print(f"Data Augmentation: +{interp['augmentation_ratio']:.1f}%")
    print(f"Data Completeness Improvement: {interp['data_completeness_improvement']:.2f}%")
    print()
    
    # Performance Summary and Recommendations
    print("📊 PERFORMANCE SUMMARY & RECOMMENDATIONS")
    print("-" * 50)
    
    # Overall grade calculation
    detection_score = min(det['detection_rate_interpolated'] / 80 * 100, 100)  # 80% is excellent
    recognition_score = rec['ocr_success_rate']
    tracking_score = min(track['avg_tracking_consistency'] / 90 * 100, 100)  # 90% is excellent
    
    overall_score = (detection_score + recognition_score + tracking_score) / 3
    
    print(f"Overall System Performance: {overall_score:.1f}/100")
    print()
    
    if overall_score >= 80:
        print("✅ EXCELLENT: System is performing very well!")
    elif overall_score >= 60:
        print("⚠️  GOOD: System performance is acceptable with room for improvement.")
    else:
        print("❌ NEEDS IMPROVEMENT: System requires optimization.")
    
    print()
    print("Recommendations:")
    if rec['ocr_success_rate'] < 70:
        print("- Consider improving OCR preprocessing (image enhancement, noise reduction)")
        print("- Evaluate license plate image quality and resolution")
    if track['avg_tracking_consistency'] < 80:
        print("- Optimize vehicle detection model for better consistency")
        print("- Consider improving tracking algorithm parameters")
    if det['detection_rate_interpolated'] < 85:
        print("- Review detection thresholds and model confidence scores")
        print("- Consider using a more robust detection model")
    
    return overall_score

def save_detailed_vehicle_stats(metrics, filename='vehicle_performance_details.csv'):
    """Save detailed per-vehicle statistics to CSV"""
    vehicle_details = metrics['tracking']['vehicle_details']
    
    df = pd.DataFrame.from_dict(vehicle_details, orient='index')
    df.index.name = 'vehicle_id'
    df = df.reset_index()
    
    df.to_csv(filename, index=False)
    print(f"📄 Detailed vehicle statistics saved to: {filename}")

def main():
    """Main function to run performance analysis"""
    print("Loading data...")
    original_data, interpolated_data = load_data()
    
    if original_data is None or interpolated_data is None:
        print("❌ Failed to load data files. Please ensure 'test.csv' and 'test_interpolated.csv' exist.")
        return
    
    print("Calculating performance metrics...")
    
    # Calculate all metrics
    metrics = {}
    metrics.update(calculate_detection_metrics(original_data, interpolated_data))
    metrics.update(calculate_recognition_metrics(interpolated_data))
    metrics.update(calculate_tracking_metrics(interpolated_data))
    metrics.update(calculate_interpolation_impact(original_data, interpolated_data))
    
    # Generate comprehensive report
    overall_score = generate_performance_report(metrics)
    
    # Save detailed statistics
    save_detailed_vehicle_stats(metrics)
    
    # Save metrics summary to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = f'performance_summary_{timestamp}.txt'
    
    with open(summary_file, 'w') as f:
        f.write(f"Performance Summary - {datetime.now()}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Overall Score: {overall_score:.1f}/100\n")
        f.write(f"Detection Rate: {metrics['detection']['detection_rate_interpolated']:.2f}%\n")
        f.write(f"OCR Success Rate: {metrics['recognition']['ocr_success_rate']:.2f}%\n")
        f.write(f"Tracking Consistency: {metrics['tracking']['avg_tracking_consistency']:.2f}%\n")
    
    print(f"📄 Performance summary saved to: {summary_file}")
    print("\n🎉 Performance analysis complete!")

if __name__ == "__main__":
    main()
