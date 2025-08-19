# src/camera_position_controller.py
# Real-time camera position controller for Bambu Lab X1 Carbon + Gimbal setup

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import serial
import socket
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import threading
import queue
import cv2

@dataclass
class CameraPosition:
    """Represents a camera position in various coordinate systems."""
    # Spherical coordinates (for analysis)
    theta: float  # Azimuth angle (radians)
    phi: float    # Inclination angle (radians)
    radius: float # Distance from origin
    
    # Cartesian coordinates (for visualization)
    x: float
    y: float
    z: float
    
    # Printer coordinates (bed-relative)
    bed_x: float  # Printer bed X position (mm)
    bed_y: float  # Printer bed Y position (mm)
    bed_z: float  # Printer bed Z position (mm)
    
    # Gimbal angles
    gimbal_pan: float   # Pan angle (degrees)
    gimbal_tilt: float  # Tilt angle (degrees)
    
    # Metadata
    priority_score: float = 0.0
    weakness_type: str = "geometry"
    estimated_capture_time: float = 0.0

class CoordinateConverter:
    """Converts between different coordinate systems for the capture setup."""
    
    def __init__(self, printer_config: Dict):
        self.printer_config = printer_config
        self.bed_center_x = printer_config.get('bed_size_x', 256) / 2
        self.bed_center_y = printer_config.get('bed_size_y', 256) / 2
        self.bed_height = printer_config.get('max_z', 256)
        
        # Object positioning on bed
        self.object_height = printer_config.get('object_height', 50)  # mm
        self.object_center_z = self.object_height / 2
        
        # Gimbal constraints
        self.gimbal_pan_range = (-180, 180)  # degrees
        self.gimbal_tilt_range = (-90, 45)   # degrees
        
        # Safety margins
        self.min_camera_distance = 100  # mm from object
        self.max_camera_distance = 300  # mm from object
    
    def spherical_to_cartesian(self, theta: float, phi: float, radius: float) -> Tuple[float, float, float]:
        """Convert spherical coordinates to Cartesian."""
        x = radius * np.cos(theta) * np.sin(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(phi)
        return x, y, z
    
    def cartesian_to_printer_coords(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Convert Cartesian coordinates to printer bed coordinates."""
        # Translate to printer coordinate system
        bed_x = self.bed_center_x + x
        bed_y = self.bed_center_y + y
        bed_z = self.object_center_z + z
        
        # Apply safety constraints
        bed_x = np.clip(bed_x, 10, self.printer_config.get('bed_size_x', 256) - 10)
        bed_y = np.clip(bed_y, 10, self.printer_config.get('bed_size_y', 256) - 10)
        bed_z = np.clip(bed_z, 10, self.bed_height - 10)
        
        return bed_x, bed_y, bed_z
    
    def calculate_gimbal_angles(self, camera_pos: Tuple[float, float, float]) -> Tuple[float, float]:
        """Calculate gimbal pan/tilt angles to look at object center."""
        x, y, z = camera_pos
        
        # Calculate angles to look at origin (object center)
        pan = np.degrees(np.arctan2(y, x))
        tilt = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
        
        # Apply gimbal constraints
        pan = np.clip(pan, self.gimbal_pan_range[0], self.gimbal_pan_range[1])
        tilt = np.clip(tilt, self.gimbal_tilt_range[0], self.gimbal_tilt_range[1])
        
        return pan, tilt
    
    def convert_weakness_to_camera_position(self, spherical_pos: Tuple[float, float], 
                                          radius: float, priority_score: float = 0.0) -> CameraPosition:
        """Convert a weak viewpoint to a complete camera position."""
        theta, phi = spherical_pos
        
        # Convert to Cartesian
        x, y, z = self.spherical_to_cartesian(theta, phi, radius)
        
        # Convert to printer coordinates
        bed_x, bed_y, bed_z = self.cartesian_to_printer_coords(x, y, z)
        
        # Calculate gimbal angles
        gimbal_pan, gimbal_tilt = self.calculate_gimbal_angles((x, y, z))
        
        return CameraPosition(
            theta=theta, phi=phi, radius=radius,
            x=x, y=y, z=z,
            bed_x=bed_x, bed_y=bed_y, bed_z=bed_z,
            gimbal_pan=gimbal_pan, gimbal_tilt=gimbal_tilt,
            priority_score=priority_score
        )

class AdaptiveViewpointGenerator:
    """Generates probe viewpoints that avoid overfitting by using different patterns."""
    
    def __init__(self, base_radius: float = 150):
        self.base_radius = base_radius
        self.used_patterns = []
        self.current_pattern = 0
        
        # Define different sampling patterns to avoid overfitting
        self.sampling_patterns = {
            'fibonacci': self._fibonacci_sphere,
            'stratified': self._stratified_sampling,
            'blue_noise': self._blue_noise_sampling,
            'adaptive': self._adaptive_sampling,
            'golden_spiral': self._golden_spiral_sampling
        }
    
    def _fibonacci_sphere(self, n_views: int) -> List[Tuple[float, float]]:
        """Original fibonacci sphere distribution."""
        points = []
        for i in range(n_views):
            phi = np.arccos(1 - 2 * (i + 0.5) / n_views)
            theta = np.pi * (1 + 5**0.5) * i
            points.append((theta, phi))
        return points
    
    def _stratified_sampling(self, n_views: int) -> List[Tuple[float, float]]:
        """Stratified sampling for more uniform distribution."""
        grid_size = int(np.sqrt(n_views))
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                # Add random jitter within each grid cell
                theta = (i + np.random.random()) * 2 * np.pi / grid_size
                phi = (j + np.random.random()) * np.pi / grid_size
                points.append((theta, phi))
                if len(points) >= n_views:
                    break
            if len(points) >= n_views:
                break
        return points[:n_views]
    
    def _blue_noise_sampling(self, n_views: int) -> List[Tuple[float, float]]:
        """Blue noise sampling for better spatial distribution."""
        points = []
        min_distance = 0.5  # Minimum distance between points
        max_attempts = 1000
        
        for _ in range(n_views):
            attempts = 0
            while attempts < max_attempts:
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                
                # Check distance to existing points
                valid = True
                for existing_theta, existing_phi in points:
                    distance = np.sqrt((theta - existing_theta)**2 + (phi - existing_phi)**2)
                    if distance < min_distance:
                        valid = False
                        break
                
                if valid:
                    points.append((theta, phi))
                    break
                attempts += 1
            
            if attempts >= max_attempts:
                # Fallback to random if can't find valid position
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                points.append((theta, phi))
        
        return points
    
    def _adaptive_sampling(self, n_views: int) -> List[Tuple[float, float]]:
        """Adaptive sampling that focuses on areas with high curvature."""
        # Start with coarse fibonacci sampling
        coarse_points = self._fibonacci_sphere(n_views // 2)
        
        # Add refinement points in areas of interest
        refined_points = []
        for i, (theta, phi) in enumerate(coarse_points):
            # Add some perturbation around each coarse point
            for _ in range(2):  # Add 2 refined points per coarse point
                perturb_theta = theta + np.random.normal(0, 0.2)
                perturb_phi = phi + np.random.normal(0, 0.1)
                # Keep phi in valid range
                perturb_phi = np.clip(perturb_phi, 0.1, np.pi - 0.1)
                refined_points.append((perturb_theta, perturb_phi))
        
        return (coarse_points + refined_points)[:n_views]
    
    def _golden_spiral_sampling(self, n_views: int) -> List[Tuple[float, float]]:
        """Golden spiral sampling with slight randomization."""
        points = []
        golden_ratio = (1 + 5**0.5) / 2
        
        for i in range(n_views):
            # Add small random perturbation to avoid exact repetition
            theta = 2 * np.pi * i / golden_ratio + np.random.normal(0, 0.1)
            phi = np.arccos(1 - 2 * i / n_views) + np.random.normal(0, 0.05)
            phi = np.clip(phi, 0.1, np.pi - 0.1)  # Keep in valid range
            points.append((theta, phi))
        
        return points
    
    def generate_probe_viewpoints(self, n_views: int, iteration: int = 0) -> List[Tuple[int, np.ndarray, Tuple[float, float]]]:
        """Generate probe viewpoints using rotating patterns to avoid overfitting."""
        # Select pattern based on iteration to ensure variety
        pattern_names = list(self.sampling_patterns.keys())
        pattern_name = pattern_names[iteration % len(pattern_names)]
        pattern_func = self.sampling_patterns[pattern_name]
        
        print(f"Using '{pattern_name}' sampling pattern for iteration {iteration}")
        
        # Generate spherical coordinates
        spherical_coords = pattern_func(n_views)
        
        # Convert to viewpoint format
        viewpoints = []
        for i, (theta, phi) in enumerate(spherical_coords):
            # Calculate camera position
            cam_pos_cartesian = np.array([
                self.base_radius * np.cos(theta) * np.sin(phi),
                self.base_radius * np.sin(theta) * np.sin(phi),
                self.base_radius * np.cos(phi)
            ])
            
            # Create look-at matrix (reuse from original code)
            cam_matrix = self._look_at(cam_pos_cartesian)
            viewpoints.append((i, cam_matrix, (theta, phi)))
        
        return viewpoints
    
    def _look_at(self, camera_pos, target_pos=np.array([0,0,0]), world_up=np.array([0, 0, 1])):
        """Create look-at matrix (from original code)."""
        forward_vector = camera_pos - target_pos
        forward_vector /= np.linalg.norm(forward_vector)
        right_vector = np.cross(world_up, forward_vector)
        right_vector /= np.linalg.norm(right_vector)
        up_vector = np.cross(forward_vector, right_vector)
        cam_to_world = np.eye(4)
        cam_to_world[:3, 0] = right_vector
        cam_to_world[:3, 1] = up_vector
        cam_to_world[:3, 2] = forward_vector
        cam_to_world[:3, 3] = camera_pos
        return cam_to_world

class PrinterController:
    """Controls the 3D printer movement via G-code commands."""
    
    def __init__(self, connection_type: str = "serial", **kwargs):
        self.connection_type = connection_type
        self.is_connected = False
        self.connection = None
        self.position_queue = queue.Queue()
        
        if connection_type == "serial":
            self.serial_port = kwargs.get('serial_port', 'COM3')
            self.baud_rate = kwargs.get('baud_rate', 115200)
        elif connection_type == "network":
            self.printer_ip = kwargs.get('printer_ip', '192.168.1.100')
            self.printer_port = kwargs.get('printer_port', 8080)
    
    def connect(self) -> bool:
        """Connect to the printer."""
        try:
            if self.connection_type == "serial":
                self.connection = serial.Serial(self.serial_port, self.baud_rate, timeout=5)
                time.sleep(2)  # Wait for connection to stabilize
            elif self.connection_type == "network":
                self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.connection.connect((self.printer_ip, self.printer_port))
            
            self.is_connected = True
            print(f"Connected to printer via {self.connection_type}")
            return True
        except Exception as e:
            print(f"Failed to connect to printer: {e}")
            return False
    
    def send_gcode(self, gcode: str) -> bool:
        """Send G-code command to printer."""
        if not self.is_connected:
            print("Printer not connected")
            return False
        
        try:
            command = f"{gcode}\n"
            if self.connection_type == "serial":
                self.connection.write(command.encode())
                response = self.connection.readline().decode().strip()
                print(f"Sent: {gcode} | Response: {response}")
            elif self.connection_type == "network":
                self.connection.send(command.encode())
            
            return True
        except Exception as e:
            print(f"Failed to send G-code: {e}")
            return False
    
    def move_to_position(self, position: CameraPosition, speed: int = 3000) -> bool:
        """Move printer to specified camera position."""
        # Move extruder to position (camera mount point)
        gcode = f"G1 X{position.bed_x:.2f} Y{position.bed_y:.2f} Z{position.bed_z:.2f} F{speed}"
        success = self.send_gcode(gcode)
        
        if success:
            # Wait for movement to complete
            self.send_gcode("M400")  # Wait for current moves to finish
            time.sleep(1)
            
            # Control gimbal (assuming servo control via M280)
            self.send_gcode(f"M280 P0 S{int(position.gimbal_pan + 90)}")  # Pan servo
            self.send_gcode(f"M280 P1 S{int(position.gimbal_tilt + 90)}")  # Tilt servo
            time.sleep(2)  # Wait for gimbal to position
        
        return success
    
    def home_printer(self) -> bool:
        """Home the printer axes."""
        return self.send_gcode("G28")  # Home all axes
    
    def disconnect(self):
        """Disconnect from printer."""
        if self.connection:
            self.connection.close()
            self.is_connected = False

class RealTimeCaptureSystem:
    """Main class coordinating the real-time capture system."""
    
    def __init__(self, printer_config: Dict):
        self.converter = CoordinateConverter(printer_config)
        self.viewpoint_generator = AdaptiveViewpointGenerator(
            base_radius=printer_config.get('capture_radius', 150)
        )
        self.printer = PrinterController(
            connection_type=printer_config.get('connection_type', 'serial'),
            **printer_config
        )
        self.iteration_count = 0
        
    def extract_camera_positions_from_weakness_analysis(self, 
                                                       weakness_results: List[Dict], 
                                                       max_positions: int = 10) -> List[CameraPosition]:
        """Extract and convert weakness analysis results to camera positions."""
        camera_positions = []
        
        for result in weakness_results[:max_positions]:
            spherical_pos = result.get('spherical_pos', (0, 0))
            priority_score = result.get('priority_score', 0.0)
            radius = self.viewpoint_generator.base_radius
            
            camera_pos = self.converter.convert_weakness_to_camera_position(
                spherical_pos, radius, priority_score
            )
            camera_pos.weakness_type = result.get('analysis', {}).get('mode', 'geometry')
            camera_positions.append(camera_pos)
        
        return camera_positions
    
    def save_camera_positions_for_printer(self, positions: List[CameraPosition], 
                                        output_path: str):
        """Save camera positions in format suitable for printer control."""
        # G-code format
        gcode_path = output_path.replace('.json', '.gcode')
        with open(gcode_path, 'w') as f:
            f.write("; Auto-generated G-code for camera positions\n")
            f.write("G28 ; Home all axes\n")
            f.write("G90 ; Absolute positioning\n")
            f.write("\n")
            
            for i, pos in enumerate(positions):
                f.write(f"; Position {i+1} - Priority: {pos.priority_score:.3f}\n")
                f.write(f"G1 X{pos.bed_x:.2f} Y{pos.bed_y:.2f} Z{pos.bed_z:.2f} F3000\n")
                f.write("M400 ; Wait for move to complete\n")
                f.write(f"M280 P0 S{int(pos.gimbal_pan + 90)} ; Pan gimbal\n")
                f.write(f"M280 P1 S{int(pos.gimbal_tilt + 90)} ; Tilt gimbal\n")
                f.write("G4 P2000 ; Wait 2 seconds\n")
                f.write("M42 P2 S255 ; Trigger camera (example pin)\n")
                f.write("G4 P1000 ; Wait 1 second\n")
                f.write("M42 P2 S0 ; Camera off\n")
                f.write("\n")
        
        # JSON format for detailed information
        json_data = {
            'iteration': self.iteration_count,
            'timestamp': time.time(),
            'positions': [
                {
                    'index': i,
                    'spherical': {'theta': pos.theta, 'phi': pos.phi, 'radius': pos.radius},
                    'cartesian': {'x': pos.x, 'y': pos.y, 'z': pos.z},
                    'printer_coords': {'x': pos.bed_x, 'y': pos.bed_y, 'z': pos.bed_z},
                    'gimbal': {'pan': pos.gimbal_pan, 'tilt': pos.gimbal_tilt},
                    'priority_score': pos.priority_score,
                    'weakness_type': pos.weakness_type
                }
                for i, pos in enumerate(positions)
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Saved {len(positions)} camera positions:")
        print(f"  G-code: {gcode_path}")
        print(f"  JSON: {output_path}")
    
    def execute_capture_sequence(self, positions: List[CameraPosition]) -> bool:
        """Execute the camera positioning and capture sequence."""
        if not self.printer.connect():
            return False
        
        try:
            # Home printer first
            self.printer.home_printer()
            
            for i, position in enumerate(positions):
                print(f"Moving to position {i+1}/{len(positions)} (Priority: {position.priority_score:.3f})")
                
                success = self.printer.move_to_position(position)
                if not success:
                    print(f"Failed to move to position {i+1}")
                    continue
                
                # Trigger camera capture here
                # This would interface with your camera system
                self.trigger_camera_capture(position, i)
                
            return True
            
        except Exception as e:
            print(f"Error during capture sequence: {e}")
            return False
        finally:
            self.printer.disconnect()
    
    def trigger_camera_capture(self, position: CameraPosition, index: int):
        """Trigger camera capture at current position."""
        # Placeholder for camera integration
        print(f"📸 Capturing image at position {index+1}")
        print(f"   Gimbal: Pan={position.gimbal_pan:.1f}°, Tilt={position.gimbal_tilt:.1f}°")
        
        # Here you would integrate with your camera system
        # cv2 capture, camera API calls, etc.
        
        time.sleep(1)  # Simulate capture time
    
    def generate_adaptive_probe_viewpoints(self, n_views: int = 128) -> List:
        """Generate probe viewpoints that change each iteration."""
        return self.viewpoint_generator.generate_probe_viewpoints(n_views, self.iteration_count)
    
    def increment_iteration(self):
        """Increment iteration counter for adaptive sampling."""
        self.iteration_count += 1


# Integration function to modify the existing weakness analysis
def integrate_with_weakness_analysis(run_path: str, printer_config_path: str):
    """Integration function to connect with existing weakness analysis system."""
    
    # Load printer configuration
    with open(printer_config_path, 'r') as f:
        printer_config = json.load(f)
    
    # Initialize real-time capture system
    capture_system = RealTimeCaptureSystem(printer_config)
    
    # Check if weakness analysis results exist
    metadata_path = Path(run_path) / "analysis" / "recommendation_metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            weakness_metadata = json.load(f)
        
        weakness_results = weakness_metadata.get('recommendations', [])
        
        # Extract camera positions
        camera_positions = capture_system.extract_camera_positions_from_weakness_analysis(
            weakness_results
        )
        
        # Save positions for printer
        positions_output = Path(run_path) / "camera_positions.json"
        capture_system.save_camera_positions_for_printer(camera_positions, str(positions_output))
        
        # Optionally execute capture sequence
        if printer_config.get('auto_capture', False):
            capture_system.execute_capture_sequence(camera_positions)
        
        return camera_positions
    else:
        print("No weakness analysis results found")
        return []


if __name__ == "__main__":
    # Example usage
    printer_config = {
        'bed_size_x': 256,
        'bed_size_y': 256, 
        'max_z': 256,
        'object_height': 50,
        'capture_radius': 150,
        'connection_type': 'serial',
        'serial_port': 'COM3',
        'baud_rate': 115200,
        'auto_capture': False
    }
    
    # Save example config
    with open('printer_config.json', 'w') as f:
        json.dump(printer_config, f, indent=2)
    
    print("Real-time capture system initialized!")
    print("Use integrate_with_weakness_analysis() to connect with your existing pipeline.")
    