import time
import cv2
from flircam import Flircam

from multiprocessing import Process, Queue


def writer(queue, filename, fps, w, h):

    framecount = 0    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(filename=filename, fourcc=fourcc, fps=fps, frameSize=(w, h))

    while True:
        frame = queue.get()
        if frame is None:
            break
        out.write(frame)
        framecount += 1
    out.release()


def main():
    
    # Initialize hand pose detector and FLIR camera
    cam = Flircam()
    # cam.start()

    # Configuration
    filename = "fixedfps_recording_1.avi"
    w, h = (720, 540)
    fps = 300.0  # <- Set your desired FPS here

    queue = Queue(maxsize=100)
    p = Process(target=writer, args=(queue, filename, fps, w, h))
    p.start()
    frame, ts, _ = cam.read_frame()

    recording = False
    recording_index = 1

    print(f"Press 'r' to start/stop recording at {fps} FPS. Press 'q' to quit.")

    start_time = None

    try:
        while True:
            # ret, frame = cap.read()
            frame, ts, _ = cam.read_frame()
            if not frame.any():
                print("Failed to grab frame.")
                break
            cv2.line(frame, (0, 400), (frame.shape[1], 400), (255,0,0), 2)

            cv2.imshow("Live Feed", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                if not recording:
                    print(f"Recording started: {filename}")
                    recording = True
                    start_time = time.time()
                else:
                    elapsed = time.time() - start_time
                    print(f"Recording stopped. Time elapsed: {elapsed:.2f}s")
                    with open(f"metadata_{filename}.txt", "w+") as f:
                        f.write(f"{elapsed:.2f}")
                    recording = False
                    recording_index += 1

            elif key == ord('q'):
                print("Quitting.")
                if recording:
                    queue.put(None)
                break
            
            cv2.putText(frame, str(ts), (300, 200), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255))
            if recording:
                if not queue.full():
                    queue.put(frame)
                # time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        queue.put(None)
        p.join()
    
        # Cleanup
        cam.cleanup()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()