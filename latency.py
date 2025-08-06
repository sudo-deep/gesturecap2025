import cv2
import time
import json
import csv
import numpy as np
from pythonosc import udp_client
from flircam import Flircam
from hand_pose_detector import HandPoseDetector

"""
Main tap-detection script with logging:
- Loads calibration results (y_line, stdev, mean)
- Detects hand taps
- Measures and logs timestamps, total latency, and processing latency
- Saves results to CSV (tableB)
"""

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
    time.sleep(1)

    osc_ip, osc_port = '127.0.0.1', 11111
    # osc_ip, osc_port = '192.168.2.2', 11111
    client = udp_client.SimpleUDPClient(osc_ip, osc_port)
    detector = HandPoseDetector()

    # Prepare CSV logging
    csv_filename = 'tableB.csv'
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp', 'internal_latency_total', 'latency_processing'])

    state = 0
    counter = 0
    print("Starting hand-tap detection. Press 'q' to exit.")

    try:
        while True:
            # Timestamp at start of frame capture
            t_start = time.time()

            # Capture frame
            # frame, ts = cam.read_frame()
            frame, ts = cam.get_frame()
            if not frame.any():
                break   

            # Measure processing latency (hand detection)
            t_proc_start = time.time()
            hands = detector.detect_hand_pose(frame)
            t_proc_end = time.time()

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
            latency_total = t_end - t_start
            latency_processing = t_proc_end - t_proc_start

            # # Log to CSV
            # csv_writer.writerow([t_start, latency_total, latency_processing])
            # csv_file.flush()

            # # Print loop duration
            # print(f"Loop time: {latency_total:.4f}s (processing: {latency_processing:.4f}s)")

    except KeyboardInterrupt:
        print("User Interrupt")

    finally:
        # Clean up
        csv_file.close()
        cam.cleanup()
        cv2.destroyAllWindows()
        print(f"Logged data to {csv_filename}")


if __name__ == '__main__':
    main_loop()
