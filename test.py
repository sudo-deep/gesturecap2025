import cv2
import numpy as np
import time
from flircam import Flircam
from hand_pose_detector import HandPoseDetector

def read_and_process_frame():
    # Setup camera and detector
    cam = Flircam()
    cam.start()
    detector = HandPoseDetector()

    try:
        # Capture frame
        frame, ts = cam.read_frame()
        if not frame.any():
            return

        # Measure processing time
        start_time = time.time()
        hands = detector.detect_hand_pose(frame)
        processing_time = time.time() - start_time

        print(f"Processing time: {processing_time:.4f} seconds")
        if hands:
            for hand in hands:
                print(f"Detected hand: {hand}")

    finally:
        # Clean up
        cam.cleanup()
