#!/usr/bin/env python3
"""
CSV Data Inspector - Verify collected data quality

Inspects CSV files from data collection to verify:
- Non-zero sensor values
- Correct number of samples
- Proper data structure
- Timestamp progression

Usage:
    python inspect_csv_data.py [csv_file]
    python inspect_csv_data.py data/button_collected/jump_*.csv
"""

import sys
import csv
from pathlib import Path
import statistics


def inspect_csv(filepath):
    """Inspect a CSV file and report on data quality"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return False
    
    print(f"\n{'='*80}")
    print(f"Inspecting: {filepath.name}")
    print(f"{'='*80}\n")
    
    # Read CSV
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("❌ File is empty!")
        return False
    
    print(f"✅ Total samples: {len(rows)}")
    
    # Check sensor types
    sensor_types = {}
    for row in rows:
        sensor = row.get('sensor', 'unknown')
        sensor_types[sensor] = sensor_types.get(sensor, 0) + 1
    
    print(f"\n📊 Sensor types:")
    for sensor, count in sorted(sensor_types.items()):
        print(f"   {sensor:25s}: {count:4d} samples")
    
    # Check for non-zero values
    accel_values = []
    gyro_values = []
    rot_values = []
    
    for row in rows:
        # Parse values
        try:
            accel_x = float(row.get('accel_x', 0.0))
            accel_y = float(row.get('accel_y', 0.0))
            accel_z = float(row.get('accel_z', 0.0))
            
            gyro_x = float(row.get('gyro_x', 0.0))
            gyro_y = float(row.get('gyro_y', 0.0))
            gyro_z = float(row.get('gyro_z', 0.0))
            
            rot_x = float(row.get('rot_x', 0.0))
            rot_y = float(row.get('rot_y', 0.0))
            rot_z = float(row.get('rot_z', 0.0))
            rot_w = float(row.get('rot_w', 1.0))
            
            # Collect non-zero values
            if abs(accel_x) > 0.001 or abs(accel_y) > 0.001 or abs(accel_z) > 0.001:
                accel_values.extend([accel_x, accel_y, accel_z])
            
            if abs(gyro_x) > 0.001 or abs(gyro_y) > 0.001 or abs(gyro_z) > 0.001:
                gyro_values.extend([gyro_x, gyro_y, gyro_z])
            
            if abs(rot_x) > 0.001 or abs(rot_y) > 0.001 or abs(rot_z) > 0.001:
                rot_values.extend([rot_x, rot_y, rot_z, rot_w])
        except (ValueError, TypeError):
            pass
    
    # Report data quality
    print(f"\n📈 Data Quality:")
    
    if accel_values:
        print(f"   Acceleration: ✅ {len(accel_values)} non-zero values")
        print(f"      Range: [{min(accel_values):.3f}, {max(accel_values):.3f}]")
        print(f"      Mean:  {statistics.mean(accel_values):.3f}")
        print(f"      Stdev: {statistics.stdev(accel_values) if len(accel_values) > 1 else 0:.3f}")
    else:
        print(f"   Acceleration: ❌ ALL ZEROS - No real data!")
    
    if gyro_values:
        print(f"   Gyroscope:    ✅ {len(gyro_values)} non-zero values")
        print(f"      Range: [{min(gyro_values):.3f}, {max(gyro_values):.3f}]")
        print(f"      Mean:  {statistics.mean(gyro_values):.3f}")
        print(f"      Stdev: {statistics.stdev(gyro_values) if len(gyro_values) > 1 else 0:.3f}")
    else:
        print(f"   Gyroscope:    ❌ ALL ZEROS - No real data!")
    
    if rot_values:
        print(f"   Rotation:     ✅ {len(rot_values)} non-zero values")
        print(f"      Range: [{min(rot_values):.3f}, {max(rot_values):.3f}]")
        print(f"      Mean:  {statistics.mean(rot_values):.3f}")
        print(f"      Stdev: {statistics.stdev(rot_values) if len(rot_values) > 1 else 0:.3f}")
    else:
        print(f"   Rotation:     ❌ ALL ZEROS - No real data!")
    
    # Check timestamps
    timestamps = []
    for row in rows:
        try:
            ts = int(row.get('timestamp', 0))
            if ts > 0:
                timestamps.append(ts)
        except (ValueError, TypeError):
            pass
    
    if timestamps:
        print(f"\n⏱  Timestamps:")
        duration_ns = timestamps[-1] - timestamps[0]
        duration_s = duration_ns / 1e9
        print(f"   Start:    {timestamps[0]}")
        print(f"   End:      {timestamps[-1]}")
        print(f"   Duration: {duration_s:.2f}s")
        if len(timestamps) > 1:
            rate = len(timestamps) / duration_s
            print(f"   Rate:     {rate:.1f} Hz")
    else:
        print(f"\n⏱  Timestamps: ❌ Missing or invalid")
    
    # Sample data
    print(f"\n📝 Sample Data (first 5 rows):")
    print(f"   {'Sensor':<25s} {'Accel X':>8s} {'Gyro X':>8s} {'Rot X':>8s}")
    print(f"   {'-'*60}")
    for i, row in enumerate(rows[:5]):
        sensor = row.get('sensor', 'unknown')
        accel_x = float(row.get('accel_x', 0.0))
        gyro_x = float(row.get('gyro_x', 0.0))
        rot_x = float(row.get('rot_x', 0.0))
        print(f"   {sensor:<25s} {accel_x:>8.3f} {gyro_x:>8.3f} {rot_x:>8.3f}")
    
    # Overall verdict
    print(f"\n🎯 Overall Verdict:")
    if accel_values or gyro_values or rot_values:
        print(f"   ✅ File contains real sensor data!")
        return True
    else:
        print(f"   ❌ File has NO real sensor data - all zeros!")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Check Watch app is streaming (toggle should be ON)")
        print(f"   2. Verify Watch app shows 'Connected!' in green")
        print(f"   3. Try running: python src/test_connection.py")
        print(f"   4. Rebuild Watch app in Android Studio")
        print(f"   5. Check sensor permissions on watch")
        return False


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python inspect_csv_data.py <csv_file>")
        print("\nExample:")
        print("  python inspect_csv_data.py data/button_collected/jump_*.csv")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Handle wildcards
    from glob import glob
    files = glob(filepath)
    
    if not files:
        print(f"❌ No files matching: {filepath}")
        sys.exit(1)
    
    # Inspect each file
    all_good = True
    for f in files:
        result = inspect_csv(f)
        all_good = all_good and result
    
    # Summary
    if len(files) > 1:
        print(f"\n{'='*80}")
        print(f"Inspected {len(files)} files")
        if all_good:
            print(f"✅ All files contain real sensor data!")
        else:
            print(f"⚠️  Some files have issues - see details above")
        print(f"{'='*80}")
    
    sys.exit(0 if all_good else 1)


if __name__ == '__main__':
    main()
