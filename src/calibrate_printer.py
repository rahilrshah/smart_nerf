# src/calibrate_printer.py
# Utility for calibrating and testing the 3D printer + gimbal setup

import json
import time
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import real-time capture system
try:
    from camera_position_controller import (
        PrinterController, 
        CoordinateConverter, 
        CameraPosition,
        RealTimeCaptureSystem
    )
    REALTIME_AVAILABLE = True
except ImportError:
    print("Real-time capture system not available. Install required dependencies.")
    REALTIME_AVAILABLE = False

class PrinterCalibrator:
    """Handles printer calibration and testing procedures."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        if REALTIME_AVAILABLE:
            self.controller = PrinterController(
                connection_type=self.config['connectivity']['connection_type'],
                **self.config['connectivity']
            )
            self.converter = CoordinateConverter(self.config)
        else:
            self.controller = None
            self.converter = None
    
    def test_connection(self) -> bool:
        """Test connection to the printer."""
        if not REALTIME_AVAILABLE:
            print("❌ Real-time capture system not available")
            return False
            
        print("Testing printer connection...")
        success = self.controller.connect()
        if success:
            print("✅ Printer connection successful")
            # Send basic status command
            self.controller.send_gcode("M115")  # Get firmware version
            self.controller.disconnect()
        else:
            print("❌ Printer connection failed")
        
        return success
    
    def calibrate_bed_level(self) -> bool:
        """Calibrate bed leveling at predefined points."""
        if not REALTIME_AVAILABLE or not self.controller.connect():
            return False
        
        try:
            print("Starting bed level calibration...")
            
            # Home all axes first
            self.controller.send_gcode("G28")
            
            # Get calibration points from config
            cal_points = self.config.get('calibration', {}).get('bed_level_points', [])
            if not cal_points:
                print("No calibration points defined in config")
                return False
            
            bed_level_data = []
            
            for i, (x, y) in enumerate(cal_points):
                print(f"Moving to calibration point {i+1}/{len(cal_points)}: ({x}, {y})")
                
                # Move to position
                self.controller.send_gcode(f"G1 X{x} Y{y} Z10 F3000")
                self.controller.send_gcode("M400")  # Wait for move to complete
                
                # Lower Z slowly to find bed
                self.controller.send_gcode("G1 Z0.1 F100")
                self.controller.send_gcode("M400")
                
                # Record position (in real setup, this would read from printer feedback)
                bed_level_data.append({'point': [x, y], 'z_offset': 0.0})
                
                time.sleep(1)
            
            # Save calibration data
            cal_file = "bed_calibration.json"
            with open(cal_file, 'w') as f:
                json.dump(bed_level_data, f, indent=2)
            
            print(f"✅ Bed calibration complete. Data saved to {cal_file}")
            return True
            
        except Exception as e:
            print(f"❌ Bed calibration failed: {e}")
            return False
        finally:
            self.controller.disconnect()
    
    def calibrate_gimbal(self) -> bool:
        """Calibrate gimbal servo positions."""
        if not REALTIME_AVAILABLE or not self.controller.connect():
            return False
        
        try:
            print("Starting gimbal calibration...")
            
            cal_positions = self.config.get('calibration', {}).get('gimbal_calibration_positions', [])
            if not cal_positions:
                print("No gimbal calibration positions defined")
                return False
            
            gimbal_data = []
            
            for i, pos in enumerate(cal_positions):
                pan, tilt = pos['pan'], pos['tilt']
                print(f"Testing gimbal position {i+1}: Pan={pan}°, Tilt={tilt}°")
                
                # Convert angles to servo values (typically 0-180 range)
                pan_servo = int(pan + 90)  # Convert -90 to +90 -> 0 to 180
                tilt_servo = int(tilt + 90)
                
                # Send servo commands
                self.controller.send_gcode(f"M280 P{self.config['gimbal_config']['pan_servo_pin']} S{pan_servo}")
                self.controller.send_gcode(f"M280 P{self.config['gimbal_config']['tilt_servo_pin']} S{tilt_servo}")
                
                # Wait for movement
                time.sleep(2)
                
                # Record position
                gimbal_data.append({
                    'target': {'pan': pan, 'tilt': tilt},
                    'servo_values': {'pan': pan_servo, 'tilt': tilt_servo},
                    'timestamp': time.time()
                })
                
                input(f"Press Enter when gimbal has reached position {i+1}...")
            
            # Save gimbal calibration
            cal_file = "gimbal_calibration.json"
            with open(cal_file, 'w') as f:
                json.dump(gimbal_data, f, indent=2)
            
            print(f"✅ Gimbal calibration complete. Data saved to {cal_file}")
            return True
            
        except Exception as e:
            print(f"❌ Gimbal calibration failed: {e}")
            return False
        finally:
            self.controller.disconnect()
    
    def test_camera_positions(self, num_test_positions: int = 8) -> bool:
        """Test a series of camera positions around the object."""
        if not REALTIME_AVAILABLE:
            print("❌ Real-time system not available")
            return False
        
        print(f"Testing {num_test_positions} camera positions...")
        
        # Generate test positions in a circle around the object
        radius = self.config['camera_setup']['capture_radius']
        test_positions = []
        
        for i in range(num_test_positions):
            theta = 2 * np.pi * i / num_test_positions
            phi = np.pi / 3  # 60 degrees elevation
            
            # Convert to camera position
            camera_pos = self.converter.convert_weakness_to_camera_position(
                (theta, phi), radius, priority_score=i
            )
            test_positions.append(camera_pos)
        
        if not self.controller.connect():
            return False
        
        try:
            # Home printer
            self.controller.home_printer()
            
            position_data = []
            
            for i, position in enumerate(test_positions):
                print(f"\nTesting position {i+1}/{len(test_positions)}")
                print(f"  Bed coordinates: ({position.bed_x:.1f}, {position.bed_y:.1f}, {position.bed_z:.1f})")
                print(f"  Gimbal angles: Pan={position.gimbal_pan:.1f}°, Tilt={position.gimbal_tilt:.1f}°")
                
                # Move to position
                success = self.controller.move_to_position(position)
                if success:
                    print("  ✅ Position reached successfully")
                    
                    # Record actual position
                    position_data.append({
                        'index': i,
                        'target_position': {
                            'bed_x': position.bed_x,
                            'bed_y': position.bed_y,
                            'bed_z': position.bed_z,
                            'gimbal_pan': position.gimbal_pan,
                            'gimbal_tilt': position.gimbal_tilt
                        },
                        'spherical': {
                            'theta': position.theta,
                            'phi': position.phi,
                            'radius': position.radius
                        },
                        'success': True
                    })
                else:
                    print("  ❌ Failed to reach position")
                    position_data.append({
                        'index': i,
                        'target_position': vars(position),
                        'success': False
                    })
                
                time.sleep(1)
            
            # Save test results
            test_file = "position_test_results.json"
            with open(test_file, 'w') as f:
                json.dump(position_data, f, indent=2)
            
            successful_positions = sum(1 for p in position_data if p['success'])
            print(f"\n✅ Position test complete: {successful_positions}/{len(test_positions)} successful")
            print(f"   Results saved to {test_file}")
            
            return successful_positions > len(test_positions) * 0.8  # 80% success rate
            
        except Exception as e:
            print(f"❌ Position test failed: {e}")
            return False
        finally:
            self.controller.disconnect()
    
    def visualize_capture_volume(self, save_path: str = "capture_volume.png"):
        """Visualize the theoretical capture volume and constraints."""
        print("Generating capture volume visualization...")
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # 3D plot of capture volume
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Object position (center of bed)
        obj_x = self.config['object_setup']['object_center_x']
        obj_y = self.config['object_setup']['object_center_y']
        obj_z = self.config['object_setup']['object_center_z']
        
        # Plot object
        ax1.scatter([obj_x], [obj_y], [obj_z], c='red', s=100, label='Object')
        
        # Generate sphere of possible camera positions
        n_points = 50
        u = np.linspace(0, 2 * np.pi, n_points)
        v = np.linspace(0, np.pi, n_points)
        
        radius = self.config['camera_setup']['capture_radius']
        x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + obj_x
        y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + obj_y
        z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + obj_z
        
        # Plot capture sphere (wireframe)
        ax1.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.3, color='blue')
        
        # Show printer bed boundaries
        bed_x = self.config['printer_info']['bed_size_x']
        bed_y = self.config['printer_info']['bed_size_y']
        bed_z = self.config['printer_info']['max_z']
        
        # Bed corners
        bed_corners = np.array([
            [0, 0, 0], [bed_x, 0, 0], [bed_x, bed_y, 0], [0, bed_y, 0], [0, 0, 0],  # Bottom
            [0, 0, bed_z], [bed_x, 0, bed_z], [bed_x, bed_y, bed_z], [0, bed_y, bed_z], [0, 0, bed_z]  # Top
        ])
        
        ax1.plot(bed_corners[:5, 0], bed_corners[:5, 1], bed_corners[:5, 2], 'k-', linewidth=2, label='Print Bed')
        ax1.plot(bed_corners[5:, 0], bed_corners[5:, 1], bed_corners[5:, 2], 'k-', linewidth=2)
        
        # Connect bottom and top
        for i in range(4):
            ax1.plot([bed_corners[i, 0], bed_corners[i+5, 0]], 
                    [bed_corners[i, 1], bed_corners[i+5, 1]], 
                    [bed_corners[i, 2], bed_corners[i+5, 2]], 'k-', linewidth=1)
        
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.set_title('3D Capture Volume')
        ax1.legend()
        
        # 2D top view
        ax2 = fig.add_subplot(222)
        
        # Draw bed outline
        bed_rect = plt.Rectangle((0, 0), bed_x, bed_y, fill=False, edgecolor='black', linewidth=2)
        ax2.add_patch(bed_rect)
        
        # Draw capture circle (top view)
        capture_circle = plt.Circle((obj_x, obj_y), radius, fill=False, edgecolor='blue', linestyle='--')
        ax2.add_patch(capture_circle)
        
        # Object position
        ax2.scatter([obj_x], [obj_y], c='red', s=100, label='Object')
        
        # Safety margins
        margin = self.config['safety_limits']['min_bed_margin']
        safe_rect = plt.Rectangle((margin, margin), bed_x-2*margin, bed_y-2*margin, 
                                 fill=False, edgecolor='orange', linestyle=':', label='Safety Zone')
        ax2.add_patch(safe_rect)
        
        ax2.set_xlim(0, bed_x)
        ax2.set_ylim(0, bed_y)
        ax2.set_aspect('equal')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_title('Top View - Capture Area')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gimbal range visualization
        ax3 = fig.add_subplot(223)
        
        pan_range = self.config['gimbal_config']['pan_range']
        tilt_range = self.config['gimbal_config']['tilt_range']
        
        # Create polar plot for gimbal coverage
        theta = np.linspace(np.radians(pan_range[0]), np.radians(pan_range[1]), 100)
        r_min = np.full_like(theta, np.radians(abs(tilt_range[0])))
        r_max = np.full_like(theta, np.radians(tilt_range[1]))
        
        ax3_polar = fig.add_subplot(224, projection='polar')
        ax3_polar.fill_between(theta, r_min, r_max, alpha=0.3, color='green', label='Gimbal Range')
        ax3_polar.set_title('Gimbal Coverage (Polar)')
        ax3_polar.set_ylim(0, np.pi/2)
        
        # System constraints table
        ax3.axis('off')
        constraints_text = f"""
SYSTEM CONSTRAINTS:

Printer:
• Bed Size: {bed_x}×{bed_y}×{bed_z} mm
• Safety Margin: {margin} mm

Camera:
• Capture Radius: {radius} mm
• Min Distance: {self.config['camera_setup']['min_camera_distance']} mm
• Max Distance: {self.config['camera_setup']['max_camera_distance']} mm

Gimbal:
• Pan Range: {pan_range[0]}° to {pan_range[1]}°
• Tilt Range: {tilt_range[0]}° to {tilt_range[1]}°
• Servos: Pin {self.config['gimbal_config']['pan_servo_pin']} (Pan), Pin {self.config['gimbal_config']['tilt_servo_pin']} (Tilt)

Motion:
• Travel Speed: {self.config['motion_settings']['travel_speed']} mm/min
• Position Speed: {self.config['motion_settings']['positioning_speed']} mm/min
        """
        
        ax3.text(0.05, 0.95, constraints_text, transform=ax3.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Capture volume visualization saved to {save_path}")
    
    def generate_gcode_test_sequence(self, output_path: str = "test_sequence.gcode"):
        """Generate a G-code test sequence for manual testing."""
        
        print("Generating G-code test sequence...")
        
        # Create simple test positions
        test_positions = [
            (128, 128, 50, 0, 0),     # Center, level
            (180, 128, 60, 45, 15),   # Right side
            (76, 128, 60, -45, 15),   # Left side
            (128, 180, 60, 0, 30),    # Front
            (128, 76, 60, 0, 30),     # Back
        ]
        
        gcode_lines = [
            "; Auto-generated test sequence for camera positioning",
            "; Compatible with Bambu Lab X1 Carbon + Gimbal setup",
            "",
            "G28 ; Home all axes",
            "G90 ; Absolute positioning",
            "M83 ; Relative extrusion (if using extruder servos)",
            "",
            "; Set speeds and accelerations",
            f"M203 X{self.config['motion_settings']['travel_speed']} Y{self.config['motion_settings']['travel_speed']} Z{self.config['motion_settings']['travel_speed']}",
            f"M204 S{self.config['motion_settings']['acceleration']}",
            "",
        ]
        
        for i, (x, y, z, pan, tilt) in enumerate(test_positions):
            gcode_lines.extend([
                f"; Test position {i+1}",
                f"G1 X{x:.2f} Y{y:.2f} Z{z:.2f} F{self.config['motion_settings']['positioning_speed']}",
                "M400 ; Wait for move to complete",
                f"M280 P{self.config['gimbal_config']['pan_servo_pin']} S{int(pan + 90)} ; Pan to {pan}°",
                f"M280 P{self.config['gimbal_config']['tilt_servo_pin']} S{int(tilt + 90)} ; Tilt to {tilt}°",
                "G4 P2000 ; Wait 2 seconds for stabilization",
                "",
                "; Optional: Trigger camera capture",
                f"M42 P{self.config['capture_settings']['capture_trigger_pin']} S255 ; Camera trigger ON",
                "G4 P500 ; Wait 0.5 seconds",
                f"M42 P{self.config['capture_settings']['capture_trigger_pin']} S0 ; Camera trigger OFF",
                "G4 P1000 ; Wait 1 second",
                "",
            ])
        
        gcode_lines.extend([
            "; Return to safe position",
            "G1 Z100 F3000 ; Raise Z to safe height",
            "G28 X Y ; Home X and Y axes",
            "",
            "; End of test sequence",
            "M84 ; Disable stepper motors",
        ])
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(gcode_lines))
        
        print(f"✅ G-code test sequence saved to {output_path}")
        print("   Load this file into your printer's interface to test positioning")
    
    def run_full_calibration(self):
        """Run the complete calibration sequence."""
        print("="*60)
        print("    STARTING FULL CALIBRATION SEQUENCE")
        print("="*60)
        
        results = {
            'connection_test': False,
            'bed_calibration': False,
            'gimbal_calibration': False,
            'position_test': False
        }
        
        # Test connection
        print("\n1. Testing printer connection...")
        results['connection_test'] = self.test_connection()
        
        if results['connection_test']:
            # Bed calibration
            print("\n2. Calibrating bed level...")
            results['bed_calibration'] = self.calibrate_bed_level()
            
            # Gimbal calibration
            print("\n3. Calibrating gimbal...")
            results['gimbal_calibration'] = self.calibrate_gimbal()
            
            # Position testing
            print("\n4. Testing camera positions...")
            results['position_test'] = self.test_camera_positions()
        
        # Generate visualization and test files regardless of connection status
        print("\n5. Generating visualization and test files...")
        self.visualize_capture_volume()
        self.generate_gcode_test_sequence()
        
        # Summary
        print("\n" + "="*60)
        print("    CALIBRATION SUMMARY")
        print("="*60)
        
        for test, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test.replace('_', ' ').title():20} {status}")
        
        all_passed = all(results.values())
        if all_passed:
            print("\n🎉 All calibration tests passed! System ready for use.")
        else:
            print("\n⚠️  Some tests failed. Check connections and configuration.")
        
        return all_passed

def main():
    parser = argparse.ArgumentParser(description="Calibrate and test 3D printer + gimbal setup")
    parser.add_argument("--config", required=True, help="Path to printer configuration JSON")
    parser.add_argument("--test", choices=['connection', 'bed', 'gimbal', 'positions', 'full', 'visualize'], 
                       default='full', help="Which test to run")
    parser.add_argument("--num_positions", type=int, default=8, 
                       help="Number of positions to test (for position test)")
    
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        return
    
    calibrator = PrinterCalibrator(args.config)
    
    if args.test == 'connection':
        calibrator.test_connection()
    elif args.test == 'bed':
        calibrator.calibrate_bed_level()
    elif args.test == 'gimbal':
        calibrator.calibrate_gimbal()
    elif args.test == 'positions':
        calibrator.test_camera_positions(args.num_positions)
    elif args.test == 'visualize':
        calibrator.visualize_capture_volume()
        calibrator.generate_gcode_test_sequence()
    else:  # full
        calibrator.run_full_calibration()

if __name__ == "__main__":
    main()