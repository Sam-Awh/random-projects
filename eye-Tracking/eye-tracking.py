"""
# Developed in a Python 3.10.x venv
"""
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import threading
import time
import json
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AccelerationConfig:
    """Configuration for cursor acceleration features"""
    # Acceleration enable/disable
    enable_acceleration: bool = True
    enable_face_movement_acceleration: bool = True
    enable_gaze_velocity_acceleration: bool = True

    # Face movement thresholds
    face_movement_threshold: float = 0.05  # Normalized face position change
    face_rotation_threshold: float = 5.0   # Degrees of face rotation
    face_acceleration_multiplier: float = 2.5  # Acceleration when face moves

    # Gaze velocity thresholds
    gaze_velocity_threshold: float = 200.0  # pixels/second
    max_gaze_velocity: float = 800.0       # pixels/second for max acceleration
    min_acceleration_factor: float = 1.0   # No acceleration
    max_acceleration_factor: float = 3.5   # Maximum acceleration

    # Directional acceleration
    large_movement_threshold: float = 100.0  # pixels in single movement
    directional_acceleration_multiplier: float = 1.8

    # Acceleration curve settings
    acceleration_curve_type: str = "linear"  # "linear", "quadratic", "cubic"
    # Smoothing factor for acceleration changes
    acceleration_smoothing: float = 0.3

    # Velocity calculation settings
    # Number of samples for velocity calculation
    velocity_window_size: int = 5
    # Maximum time gap for velocity calc (seconds)
    velocity_time_threshold: float = 0.1


@dataclass
class EyeTrackingConfig:
    """Enhanced configuration class with acceleration settings"""
    # Camera settings
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30

    # MediaPipe settings
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    refine_landmarks: bool = True

    # Eye tracking parameters
    iris_landmarks: Tuple[int, ...] = (468, 469, 470, 471, 472)  # Left iris
    right_iris_landmarks: Tuple[int, ...] = (
        473, 474, 475, 476, 477)  # Right iris
    left_eye_landmarks: Tuple[int, int] = (145, 159)
    blink_threshold: float = 0.004

    # Mouse control settings
    sensitivity_x: float = 1.0
    sensitivity_y: float = 1.0
    smoothing_factor: float = 0.3
    click_delay: float = 0.1
    dwell_time: float = 1.0

    # UI settings
    show_landmarks: bool = True
    show_fps: bool = True
    show_acceleration_debug: bool = True

    # Acceleration configuration
    acceleration: AccelerationConfig = AccelerationConfig()

    @classmethod
    def load_from_file(cls, filename: str) -> 'EyeTrackingConfig':
        """Load configuration from JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            # Handle nested acceleration config
            if 'acceleration' in data:
                acc_data = data.pop('acceleration')
                config = cls(**data)
                config.acceleration = AccelerationConfig(**acc_data)
                return config
            return cls(**data)
        except FileNotFoundError:
            logger.warning(f"Config file {filename} not found, using defaults")
            return cls()

    def save_to_file(self, filename: str) -> None:
        """Save configuration to JSON file"""
        data = self.__dict__.copy()
        data['acceleration'] = self.acceleration.__dict__
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


class VelocityTracker:
    """Track gaze velocity for acceleration calculations"""

    def __init__(self, config: AccelerationConfig):
        self.config = config
        self.position_history = deque(maxlen=config.velocity_window_size)
        self.time_history = deque(maxlen=config.velocity_window_size)
        self.current_velocity = 0.0

    def add_position(self, x: float, y: float) -> None:
        """Add new gaze position and calculate velocity"""
        current_time = time.time()

        self.position_history.append((x, y))
        self.time_history.append(current_time)

        if len(self.position_history) >= 2:
            self.current_velocity = self._calculate_velocity()

    def _calculate_velocity(self) -> float:
        """Calculate current gaze velocity in pixels/second"""
        if len(self.position_history) < 2:
            return 0.0

        # Use recent positions within time threshold
        recent_positions = []
        recent_times = []
        current_time = self.time_history[-1]

        for i in range(len(self.position_history) - 1, -1, -1):
            time_diff = current_time - self.time_history[i]
            if time_diff <= self.config.velocity_time_threshold:
                recent_positions.append(self.position_history[i])
                recent_times.append(self.time_history[i])
            else:
                break

        if len(recent_positions) < 2:
            return 0.0

        # Calculate distance and time difference
        pos1 = recent_positions[-1]  # Most recent
        pos2 = recent_positions[0]   # Oldest in range
        time1 = recent_times[-1]
        time2 = recent_times[0]

        distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        time_diff = time1 - time2

        if time_diff > 0:
            velocity = distance / time_diff
            return velocity

        return 0.0

    def get_velocity(self) -> float:
        """Get current velocity"""
        return self.current_velocity


class FaceMovementTracker:
    """Track face position and rotation for acceleration triggers"""

    def __init__(self, config: AccelerationConfig):
        self.config = config
        self.last_face_center = None
        self.last_face_rotation = None
        self.face_movement_detected = False

    def update_face_data(self, landmarks, frame_shape) -> None:
        """Update face position and detect significant movement"""
        try:
            frame_h, frame_w = frame_shape[:2]

            # Calculate face center using key landmarks
            face_landmarks = [33, 263, 1, 61, 291, 199]  # Key face points
            center_x = sum(
                landmarks[i].x for i in face_landmarks) / len(face_landmarks)
            center_y = sum(
                landmarks[i].y for i in face_landmarks) / len(face_landmarks)

            current_center = (center_x, center_y)

            # Calculate approximate face rotation using eye positions
            left_eye = landmarks[33]   # Left eye outer corner
            right_eye = landmarks[263]  # Right eye outer corner

            eye_vector = (right_eye.x - left_eye.x, right_eye.y - left_eye.y)
            current_rotation = math.atan2(
                eye_vector[1], eye_vector[0]) * 180 / math.pi

            # Detect significant movement
            self.face_movement_detected = False

            if self.last_face_center is not None:
                # Check position change
                pos_change = math.sqrt(
                    (current_center[0] - self.last_face_center[0])**2 +
                    (current_center[1] - self.last_face_center[1])**2
                )

                if pos_change > self.config.face_movement_threshold:
                    self.face_movement_detected = True
                    logger.debug(f"Face movement detected: {pos_change:.4f}")

            if self.last_face_rotation is not None:
                # Check rotation change
                rotation_change = abs(
                    current_rotation - self.last_face_rotation)
                # Handle angle wrapping
                if rotation_change > 180:
                    rotation_change = 360 - rotation_change

                if rotation_change > self.config.face_rotation_threshold:
                    self.face_movement_detected = True
                    logger.debug(
                        f"Face rotation detected: {rotation_change:.2f}°")

            # Update history
            self.last_face_center = current_center
            self.last_face_rotation = current_rotation

        except Exception as e:
            logger.error(f"Error tracking face movement: {e}")

    def is_face_moving(self) -> bool:
        """Check if significant face movement was detected"""
        return self.face_movement_detected


class AccelerationCalculator:
    """Calculate cursor acceleration based on various factors"""

    def __init__(self, config: AccelerationConfig):
        self.config = config
        self.last_acceleration = 1.0

    def calculate_acceleration_factor(self,
                                      velocity: float,
                                      face_moving: bool,
                                      movement_distance: float) -> float:
        """Calculate acceleration factor based on multiple inputs"""
        if not self.config.enable_acceleration:
            return 1.0

        acceleration_factors = []

        # 1. Gaze velocity-based acceleration
        if self.config.enable_gaze_velocity_acceleration:
            velocity_factor = self._calculate_velocity_acceleration(velocity)
            acceleration_factors.append(velocity_factor)

        # 2. Face movement-based acceleration
        if self.config.enable_face_movement_acceleration and face_moving:
            face_factor = self.config.face_acceleration_multiplier
            acceleration_factors.append(face_factor)
            logger.debug(f"Face movement acceleration: {face_factor:.2f}")

        # 3. Large directional movement acceleration
        if movement_distance > self.config.large_movement_threshold:
            directional_factor = self.config.directional_acceleration_multiplier
            acceleration_factors.append(directional_factor)
            logger.debug(
                f"Large movement acceleration: {directional_factor:.2f}")

        # Combine acceleration factors (use maximum)
        if acceleration_factors:
            combined_factor = max(acceleration_factors)
        else:
            combined_factor = self.config.min_acceleration_factor

        # Clamp to configured range
        combined_factor = max(self.config.min_acceleration_factor,
                              min(self.config.max_acceleration_factor, combined_factor))

        # Apply smoothing
        smoothed_factor = (self.last_acceleration * (1 - self.config.acceleration_smoothing) +
                           combined_factor * self.config.acceleration_smoothing)

        self.last_acceleration = smoothed_factor
        return smoothed_factor

    def _calculate_velocity_acceleration(self, velocity: float) -> float:
        """Calculate acceleration based on gaze velocity"""
        if velocity <= self.config.gaze_velocity_threshold:
            return self.config.min_acceleration_factor

        if velocity >= self.config.max_gaze_velocity:
            return self.config.max_acceleration_factor

        # Normalize velocity to 0-1 range
        normalized_velocity = ((velocity - self.config.gaze_velocity_threshold) /
                               (self.config.max_gaze_velocity - self.config.gaze_velocity_threshold))

        # Apply acceleration curve
        if self.config.acceleration_curve_type == "linear":
            curve_value = normalized_velocity
        elif self.config.acceleration_curve_type == "quadratic":
            curve_value = normalized_velocity ** 2
        elif self.config.acceleration_curve_type == "cubic":
            curve_value = normalized_velocity ** 3
        else:
            curve_value = normalized_velocity

        # Map to acceleration range
        acceleration_range = self.config.max_acceleration_factor - \
            self.config.min_acceleration_factor
        acceleration_factor = self.config.min_acceleration_factor + \
            (curve_value * acceleration_range)

        return acceleration_factor


class AcceleratedMouseController:
    """Enhanced mouse controller with acceleration capabilities"""

    def __init__(self, config: EyeTrackingConfig):
        self.config = config
        self.velocity_tracker = VelocityTracker(config.acceleration)
        self.face_tracker = FaceMovementTracker(config.acceleration)
        self.acceleration_calc = AccelerationCalculator(config.acceleration)

        self.last_position: Optional[Tuple[float, float]] = None
        self.screen_w, self.screen_h = pyautogui.size()

        # Disable pyautogui safety features for smooth movement
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = True

        # Debug tracking
        self.debug_info = {
            'velocity': 0.0,
            'acceleration_factor': 1.0,
            'face_moving': False,
            'movement_distance': 0.0
        }

    def update_face_tracking(self, landmarks, frame_shape) -> None:
        """Update face movement tracking"""
        self.face_tracker.update_face_data(landmarks, frame_shape)

    def move_with_acceleration(self, x: float, y: float) -> None:
        """Move mouse with intelligent acceleration"""
        # Update velocity tracking
        self.velocity_tracker.add_position(x, y)

        # Calculate base target position
        target_x = max(0, min(self.screen_w - 1, x *
                       self.config.sensitivity_x))
        target_y = max(0, min(self.screen_h - 1, y *
                       self.config.sensitivity_y))

        if self.last_position is None:
            pyautogui.moveTo(target_x, target_y)
            self.last_position = (target_x, target_y)
            return

        # Calculate movement distance
        last_x, last_y = self.last_position
        movement_distance = math.sqrt(
            (target_x - last_x)**2 + (target_y - last_y)**2)

        # Get current velocity
        current_velocity = self.velocity_tracker.get_velocity()

        # Check if face is moving
        face_moving = self.face_tracker.is_face_moving()

        # Calculate acceleration factor
        acceleration_factor = self.acceleration_calc.calculate_acceleration_factor(
            velocity=current_velocity,
            face_moving=face_moving,
            movement_distance=movement_distance
        )

        # Apply acceleration to movement
        if acceleration_factor > 1.0:
            # Accelerate movement
            accelerated_x = last_x + (target_x - last_x) * acceleration_factor
            accelerated_y = last_y + (target_y - last_y) * acceleration_factor

            # Clamp to screen bounds
            accelerated_x = max(0, min(self.screen_w - 1, accelerated_x))
            accelerated_y = max(0, min(self.screen_h - 1, accelerated_y))

            final_x = accelerated_x
            final_y = accelerated_y
        else:
            final_x = target_x
            final_y = target_y

        # Apply smoothing
        smooth_x = last_x + (final_x - last_x) * self.config.smoothing_factor
        smooth_y = last_y + (final_y - last_y) * self.config.smoothing_factor

        # Update debug info
        self.debug_info.update({
            'velocity': current_velocity,
            'acceleration_factor': acceleration_factor,
            'face_moving': face_moving,
            'movement_distance': movement_distance
        })

        try:
            pyautogui.moveTo(smooth_x, smooth_y)
            self.last_position = (smooth_x, smooth_y)

            if acceleration_factor > 1.1:  # Log significant acceleration
                logger.debug(f"Acceleration applied: {acceleration_factor:.2f}x "
                             f"(velocity: {current_velocity:.1f}, face: {face_moving})")

        except Exception as e:
            logger.error(f"Mouse movement error: {e}")

    def get_debug_info(self) -> Dict:
        """Get current debug information"""
        return self.debug_info.copy()

    def click(self) -> None:
        """Perform mouse click"""
        try:
            pyautogui.click()
            time.sleep(self.config.click_delay)
        except Exception as e:
            logger.error(f"Mouse click error: {e}")


class EnhancedFaceTracker:
    """Enhanced face tracker with acceleration support"""

    def __init__(self, config: EyeTrackingConfig):
        self.config = config
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=config.refine_landmarks,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Tracking state
        self.face_detected = False
        self.last_landmarks = None

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Process frame and extract facial landmarks with acceleration data"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                landmarks = results.multi_face_landmarks[0].landmark

                if len(landmarks) < 468:
                    logger.warning(f"Insufficient landmarks: {len(landmarks)}")
                    return frame, None

                self.face_detected = True
                self.last_landmarks = landmarks

                # Extract eye tracking data
                eye_data = self._extract_eye_data_safe(landmarks, frame.shape)

                # Add landmark data for face movement tracking
                if eye_data:
                    eye_data['landmarks'] = landmarks

                # Draw landmarks if enabled
                if self.config.show_landmarks and eye_data:
                    frame = self._draw_landmarks_safe(frame, landmarks)

                return frame, eye_data
            else:
                self.face_detected = False
                return frame, None

        except Exception as e:
            logger.error(f"Face tracking error: {e}")
            return frame, None

    def _extract_eye_data_safe(self, landmarks, frame_shape) -> Optional[Dict]:
        """Safely extract eye tracking data"""
        try:
            frame_h, frame_w = frame_shape[:2]
            total_landmarks = len(landmarks)

            if total_landmarks < 468:
                return None

            # Get iris center or fallback
            if total_landmarks >= 478:
                iris_center = landmarks[473]  # Right iris center
            else:
                # Fallback to eye center estimation
                left_corner = landmarks[33]
                right_corner = landmarks[133]
                iris_center = type('obj', (object,), {
                    'x': (left_corner.x + right_corner.x) / 2,
                    'y': (left_corner.y + right_corner.y) / 2,
                    'z': (left_corner.z + right_corner.z) / 2
                })()

            iris_x = iris_center.x * frame_w
            iris_y = iris_center.y * frame_h

            # Calculate blink ratio
            blink_ratio = self._calculate_blink_ratio_safe(landmarks)

            return {
                'iris_position': (iris_x, iris_y),
                'screen_position': (iris_center.x * pyautogui.size()[0],
                                    iris_center.y * pyautogui.size()[1]),
                'blink_ratio': blink_ratio,
                'is_blink': blink_ratio < self.config.blink_threshold if blink_ratio is not None else False,
                'confidence': 1.0,
                'total_landmarks': total_landmarks
            }

        except Exception as e:
            logger.error(f"Error extracting eye data: {e}")
            return None

    def _calculate_blink_ratio_safe(self, landmarks) -> Optional[float]:
        """Calculate blink ratio safely"""
        try:
            if len(landmarks) > max(159, 145):
                return abs(landmarks[159].y - landmarks[145].y)
            return None
        except Exception as e:
            logger.error(f"Error calculating blink ratio: {e}")
            return None

    def _draw_landmarks_safe(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Draw landmarks safely"""
        try:
            frame_h, frame_w = frame.shape[:2]

            # Draw basic eye landmarks
            basic_landmarks = [33, 133, 159, 145, 362, 398]
            for idx in basic_landmarks:
                if idx < len(landmarks):
                    landmark = landmarks[idx]
                    x = int(landmark.x * frame_w)
                    y = int(landmark.y * frame_h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Draw iris landmarks if available
            if len(landmarks) >= 478:
                for idx in range(468, 478):
                    landmark = landmarks[idx]
                    x = int(landmark.x * frame_w)
                    y = int(landmark.y * frame_h)
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            return frame
        except Exception as e:
            logger.error(f"Error drawing landmarks: {e}")
            return frame


class AcceleratedEyeMouseSystem:
    """Main system with acceleration capabilities"""

    def __init__(self, config_file: str = "eye_config.json"):
        self.config = EyeTrackingConfig.load_from_file(config_file)
        self.mouse_controller = AcceleratedMouseController(self.config)
        self.face_tracker = EnhancedFaceTracker(self.config)

        # System state
        self.running = False
        self.paused = False
        self.camera = None
        self.last_blink_time = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        # Threading
        self.capture_thread = None
        self.frame_queue = []
        self.frame_lock = threading.Lock()

    def initialize_camera(self) -> bool:
        """Initialize camera"""
        try:
            self.camera = cv2.VideoCapture(self.config.camera_index)
            if not self.camera.isOpened():
                logger.error("Failed to open camera")
                return False

            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT,
                            self.config.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.config.target_fps)

            logger.info("Camera initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            return False

    def capture_frames(self) -> None:
        """Capture frames in separate thread"""
        while self.running and self.camera and self.camera.isOpened():
            try:
                ret, frame = self.camera.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    with self.frame_lock:
                        self.frame_queue.append(frame)
                        if len(self.frame_queue) > 2:
                            self.frame_queue.pop(0)
                else:
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"Frame capture error: {e}")
                time.sleep(0.1)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get latest frame"""
        with self.frame_lock:
            if self.frame_queue:
                return self.frame_queue.pop(-1)
        return None

    def calculate_fps(self) -> None:
        """Calculate FPS"""
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time

    def draw_ui_elements(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI with acceleration debug info"""
        try:
            if self.config.show_fps:
                cv2.putText(frame, f"FPS: {self.current_fps}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Status indicators
            status_color = (
                0, 255, 0) if self.face_tracker.face_detected else (0, 0, 255)
            status_text = "Face Detected" if self.face_tracker.face_detected else "No Face"
            cv2.putText(frame, status_text, (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            # Acceleration debug info
            if self.config.show_acceleration_debug:
                debug_info = self.mouse_controller.get_debug_info()
                y_offset = 60

                cv2.putText(frame, f"Velocity: {debug_info['velocity']:.1f} px/s",
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

                cv2.putText(frame, f"Acceleration: {debug_info['acceleration_factor']:.2f}x",
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

                if debug_info['face_moving']:
                    cv2.putText(frame, "FACE MOVEMENT DETECTED",
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if self.paused:
                cv2.putText(frame, "PAUSED (Press SPACE to resume)", (10, frame.shape[0] - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        except Exception as e:
            logger.error(f"Error drawing UI: {e}")

        return frame

    def handle_eye_data(self, eye_data: Dict) -> None:
        """Process eye data with acceleration"""
        try:
            if not eye_data or self.paused:
                return

            # Update face movement tracking
            if 'landmarks' in eye_data:
                self.mouse_controller.update_face_tracking(
                    eye_data['landmarks'],
                    (self.config.frame_height, self.config.frame_width)
                )

            # Move mouse with acceleration
            screen_x, screen_y = eye_data['screen_position']
            self.mouse_controller.move_with_acceleration(screen_x, screen_y)

            # Handle blink clicking
            if eye_data['is_blink']:
                current_time = time.time()
                if current_time - self.last_blink_time > self.config.dwell_time:
                    self.mouse_controller.click()
                    self.last_blink_time = current_time
                    logger.info("Blink click detected")

        except Exception as e:
            logger.error(f"Error handling eye data: {e}")

    def handle_keyboard_input(self) -> bool:
        """Handle keyboard input"""
        try:
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                return False
            elif key == ord(' '):  # Space - pause/resume
                self.paused = not self.paused
                logger.info(f"System {'paused' if self.paused else 'resumed'}")
            elif key == ord('a'):  # A - toggle acceleration
                self.config.acceleration.enable_acceleration = not self.config.acceleration.enable_acceleration
                logger.info(
                    f"Acceleration {'enabled' if self.config.acceleration.enable_acceleration else 'disabled'}")
            elif key == ord('f'):  # F - toggle face movement acceleration
                self.config.acceleration.enable_face_movement_acceleration = not self.config.acceleration.enable_face_movement_acceleration
                logger.info(
                    f"Face movement acceleration {'enabled' if self.config.acceleration.enable_face_movement_acceleration else 'disabled'}")
            elif key == ord('v'):  # V - toggle velocity acceleration
                self.config.acceleration.enable_gaze_velocity_acceleration = not self.config.acceleration.enable_gaze_velocity_acceleration
                logger.info(
                    f"Velocity acceleration {'enabled' if self.config.acceleration.enable_gaze_velocity_acceleration else 'disabled'}")
            elif key == ord('s'):  # S - save config
                self.config.save_to_file("eye_config.json")
                logger.info("Configuration saved")

            return True

        except Exception as e:
            logger.error(f"Error handling keyboard: {e}")
            return True

    def cleanup(self) -> None:
        """Clean up resources"""
        logger.info("Shutting down accelerated eye tracking system...")
        self.running = False

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)

        if self.camera:
            self.camera.release()

        cv2.destroyAllWindows()
        logger.info("Cleanup completed")

    def run(self) -> None:
        """Main execution loop"""
        logger.info("Starting Accelerated Eye Controlled Mouse System")

        if not self.initialize_camera():
            return

        self.running = True
        self.capture_thread = threading.Thread(
            target=self.capture_frames, daemon=True)
        self.capture_thread.start()

        logger.info("Controls:")
        logger.info("  SPACE - Pause/Resume")
        logger.info("  A - Toggle Acceleration")
        logger.info("  F - Toggle Face Movement Acceleration")
        logger.info("  V - Toggle Velocity Acceleration")
        logger.info("  S - Save Configuration")
        logger.info("  ESC - Exit")

        try:
            while self.running:
                frame = self.get_latest_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                processed_frame, eye_data = self.face_tracker.process_frame(
                    frame)

                if eye_data and not self.paused:
                    self.handle_eye_data(eye_data)

                processed_frame = self.draw_ui_elements(processed_frame)
                cv2.imshow('Accelerated Eye Controlled Mouse', processed_frame)

                self.calculate_fps()

                if not self.handle_keyboard_input():
                    break

                time.sleep(1.0 / self.config.target_fps)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            self.cleanup()


def main():
    """Main entry point"""
    try:
        system = AcceleratedEyeMouseSystem("eye_config.json")
        system.run()
    except Exception as e:
        logger.error(f"System error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
