import cv2
from collections import deque
from flircam import Flircam


def main():
    # Initialize FLIR camera
    cam = Flircam()
    # cam.start()

    ts_history = deque(maxlen=30)  # keep last 30 timestamps for rolling avg

    print("Press 'q' to quit.")

    try:
        while True:
            frame, ts, _ = cam.read_frame()
            if frame is None or not frame.any():
                print("Failed to grab frame.")
                break

            # Add timestamp to history
            ts_history.append(ts)

            # Compute rolling FPS
            fps = 0.0
            if len(ts_history) > 1:
                elapsed = ts_history[-1] - ts_history[0]
                if elapsed > 0:
                    fps = (len(ts_history) - 1) / elapsed

            print(fps)
            # Draw timestamp and FPS
            cv2.putText(frame, f"ts: {ts:.6f}s", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.2f}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Live Feed", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting.")
                break

    except KeyboardInterrupt:
        pass
    finally:
        cam.cleanup()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
