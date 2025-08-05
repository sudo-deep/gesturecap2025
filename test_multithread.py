import cv2
import time
import json
import csv
import numpy as np
import threading
from collections import deque
from pythonosc import udp_client
from flircam import Flircam
from hand_pose_detector import HandPoseDetector

"""
Main tap-detection script with threaded frame capture:
- Loads calibration results (y_line, stdev, mean)
- Uses separate thread for frame capture with thread-safe buffer
- Detects hand taps with reduced latency
- Measures and logs timestamps, total latency, and processing latency
- Saves results to CSV (tableB)
"""

class ThreadedFrameCapture:
    def __init__(self, camera, buffer_size=1):
        self.camera = camera
        self.buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        
    def start(self):
        """Start the frame capture thread"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the frame capture thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
            
    def _capture_frames(self):
        """Internal method to continuously capture frames"""
        while self.running:
            try:
                frame, ts = self.camera.read_frame()
                if frame.any():
                    with self.lock:
                        # Add timestamp when frame was captured
                        self.buffer.append((frame, ts, time.time()))
                else:
                    # Small delay if no frame available
                    time.sleep(0.001)
            except Exception as e:
                print(f"Frame capture error: {e}")
                time.sleep(0.001)
                
    def get_latest_frame(self):
        """Get the most recent frame from buffer"""
        with self.lock:
            if self.buffer:
                return self.buffer[-1]  # Return (frame, original_ts, capture_time)
            return None, None, None
            
    def get_buffer_size(self):
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)

def load_calibration(calib_file='calibration.json'):
    with open(calib_file, 'r') as fp:
        data = json.load(fp)
    return data['y_line'], data['std_offset'], data['mean_offset']

def main_loop():
    # Load calibration
    y_line, stdev, mean = load_calibration()
    threshold = mean + 3 * stdev
    print(f"Using y_line={y_line}, threshold={threshold:.2f}px")
    
    # Setup camera, OSC, detector
    cam = Flircam()
    cam.start()
    
    # Initialize threaded frame capture
    frame_capture = ThreadedFrameCapture(cam, buffer_size=1)
    frame_capture.start()
    
    # Wait a moment for buffer to fill
    time.sleep(0.1)
    
    osc_ip, osc_port = '127.0.0.1', 11111
    # osc_ip, osc_port = '192.168.2.2', 11111
    client = udp_client.SimpleUDPClient(osc_ip, osc_port)
    detector = HandPoseDetector()
    
    # Prepare CSV logging
    csv_filename = 'tableB.csv'
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp', 'internal_latency_total', 'latency_processing', 'frame_age'])
    
    state = 0
    counter = 0
    print("Starting hand-tap detection. Press 'q' to exit.")
    
    try:
        while True:
            # Timestamp at start of processing
            t_start = time.time()
            
            # Get latest frame from buffer (non-blocking)
            frame, _, capture_time = frame_capture.get_latest_frame()
            
            if frame is None:
                # No frame available, continue
                time.sleep(0.001)
                continue
                
            # Calculate frame age (how old the frame is)
            frame_age = t_start - capture_time
            
            # Measure processing latency (hand detection)
            t_proc_start = time.time()
            hands = detector.detect_hand_pose(frame)
            t_proc_end = time.time()
            
            # Timestamp at end
            t_end = time.time()
            
            # Compute latencies
            latency_total = t_end - t_start
            latency_processing = t_proc_end - t_proc_start

            tapped = False
            if hands:
                for hand in hands:
                    # Skip right hand (use left)
                    if hand.get('label', '').lower() == 'right':
                        continue
                    
                    # Compute average y of fingertips
                    ys = [hand['landmarks'].landmark[i].y * frame.shape[0] for i in range(17, 21)]
                    avg_y = np.mean(ys)
                    dist = abs(avg_y - y_line)
                    
                    # Tap state machine
                    if dist >= threshold and state == 1:
                        state = 0
                    elif dist < threshold and state == 0:
                        state = 1
                        tapped = True
                        counter += 1
                        print(f"Tap #{counter}")
                        client.send_message('/trigger', 1)
            
            # Timestamp at end
            t_end = time.time()
            
            # Compute latencies
            frame_age = t_end - capture_time
            latency_total = t_end - t_start
            latency_processing = t_proc_end - t_proc_start
            
            # # Log to CSV (including frame age for analysis)
            # csv_writer.writerow([t_start, latency_total, latency_processing, frame_age])
            # csv_file.flush()
            
            # Optional: Print performance metrics
            # buffer_size = frame_capture.get_buffer_size()
            # print(f"Loop: {latency_total:.4f}s, Proc: {latency_processing:.4f}s, "
            #       f"Frame age: {frame_age:.4f}s, Buffer: {buffer_size}")
            
    except KeyboardInterrupt:
        print("User Interrupt")
    finally:
        # Clean up
        frame_capture.stop()
        csv_file.close()
        cam.cleanup()
        cv2.destroyAllWindows()
        print(f"Logged data to {csv_filename}")

if __name__ == '__main__':
    main_loop()