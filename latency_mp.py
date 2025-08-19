from multiprocessing import Lock, shared_memory, Event, Process, Value
import multiprocessing as mp
import time
import numpy as np
import json
import csv
import os
from datetime import datetime
import shutil
from pythonosc import udp_client
from flircam import Flircam
from hand_pose_detector import HandPoseDetector
import matplotlib.pyplot as plt


def load_calibration(calib_file='calibration.json'):
    with open(calib_file, 'r') as fp:
        data = json.load(fp)
    return data['y_line'], data['std_offset'], data['mean_offset']


def precise_sleep(target_duration):
    end_time = time.perf_counter() + target_duration
    while time.perf_counter() < end_time:
        pass


FRAME_SHAPE = (540, 720, 3)
FRAME_DTYPE = np.uint8
TEST_IMG = np.zeros(FRAME_SHAPE, dtype=FRAME_DTYPE)

    
def producer(shm_name0, shm_name1, cur_idx, stop_event, ts_value,
             t_read_total_v, t_frameacq_v, t_getts_v, t_frameconv_v):

    cam = Flircam()
    # cam.start()
    # time.sleep(1)
    shm0 = shared_memory.SharedMemory(name=shm_name0)
    shm1 = shared_memory.SharedMemory(name=shm_name1)
    buf0 = np.ndarray(FRAME_SHAPE, dtype=FRAME_DTYPE, buffer=shm0.buf)
    buf1 = np.ndarray(FRAME_SHAPE, dtype=FRAME_DTYPE, buffer=shm1.buf)

    try:
        while not stop_event.is_set():
            # Time the total read_frame call
            t_start = time.perf_counter()
            # Expectation: cam.read_frame() returns (frame, cam_ts, (t_frameacq, t_getts, t_frameconv))
            frame, cam_ts_inner, (t_frameacq, t_getts, t_frameconv) = cam.read_frame()
            t_end = time.perf_counter()
            t_total = t_end - t_start  # seconds

            if not frame.any():
                stop_event.set()
                break

            # Choose the non-current buffer to write to
            write_idx = 1 - cur_idx.value

            if write_idx == 0:
                np.copyto(buf0, frame)
            else:
                np.copyto(buf1, frame)

            # Publish timing values (store in shared Values for consumer to read)
            t_read_total_v.value = t_total
            t_frameacq_v.value = t_frameacq
            t_getts_v.value = t_getts
            t_frameconv_v.value = t_frameconv

            # Publish timestamp of when frame became available (use perf_counter for accuracy)
            ts_value.value = t_end

            # Publish index last written
            cur_idx.value = write_idx

            # small sleep if desired
            # precise_sleep(0.003)
    except KeyboardInterrupt:
        print("PRODUCER: KeyboardInterrupt")
    finally:
        cam.cleanup()
        shm0.close()
        shm1.close()
        print("PRODUCER EXITS GRACEFULLY")


def consumer(shm_name0, shm_name1, cur_idx, stop_event, ts_value,
             t_read_total_v, t_frameacq_v, t_getts_v, t_frameconv_v):
    """
    Consumer: reads latest frame, measures detect_hand_pose() time, and on each OSC trigger
    appends a CSV row with:
      [record_time, tap_number, frame_age_ms, t_read_total_ms, t_frameacq_ms, t_getts_ms, t_frameconv_ms, detect_time_ms]

    Creates a new archival CSV named `tableB_YYYYmmdd_HHMMSS.csv` each run and also writes/updates
    a copy named `tableB.csv` (always overwritten to match the latest run).
    """

    y_line, stdev, mean = load_calibration()
    threshold = mean + 3 * stdev
    print(f"Using y_line={y_line}, threshold={threshold:.2f}px")

    osc_ip, osc_port = '127.0.0.1', 11111
    # osc_ip, osc_port = '192.168.2.3', 5005
    client = udp_client.SimpleUDPClient(osc_ip, osc_port)

    detector = HandPoseDetector()

    shm0 = shared_memory.SharedMemory(name=shm_name0)
    shm1 = shared_memory.SharedMemory(name=shm_name1)
    buf0 = np.ndarray(FRAME_SHAPE, dtype=FRAME_DTYPE, buffer=shm0.buf)
    buf1 = np.ndarray(FRAME_SHAPE, dtype=FRAME_DTYPE, buffer=shm1.buf)

    # Prepare CSV filenames: archival + fixed name
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_csv = f"tableB_{now_str}.csv"
    fixed_csv = 'tableB.csv'

    header = ['record_time_perf', 'tap_number', 'frame_age_ms',
              't_read_total_ms', 't_frameacq_ms', 't_getts_ms', 't_frameconv_ms',
              'detect_time_ms']

    # # Create archival file and write header
    # with open(archive_csv, 'w', newline='') as af:
    #     writer = csv.writer(af)
    #     writer.writerow(header)

    # Create/overwrite fixed file with header (so tableB.csv is fresh each run)
    with open(fixed_csv, 'w', newline='') as ff:
        writer = csv.writer(ff)
        writer.writerow(header)

    def log_row_to_csv(row):
        # Append to both files so you keep an archival copy and a stable filename
        with open(archive_csv, 'a', newline='') as af, open(fixed_csv, 'a', newline='') as ff:
            # writer_a = csv.writer(af)
            writer_f = csv.writer(ff)
            # writer_a.writerow(row)
            writer_f.writerow(row)

    time.sleep(0.5)  # Let producer warm up

    state = 0
    counter = 0
    print("Starting hand-tap detection. Press 'q' to exit (if you implement a UI).")

    try:
        while not stop_event.is_set():
            read_idx = cur_idx.value  # Read latest published index

            if read_idx == 0:
                frame = buf0.copy()
            else:
                frame = buf1.copy()

            # Frame age in milliseconds
            frame_age_ms = (time.perf_counter() - ts_value.value) * 1000.0

            # Read the timing breakdowns published by producer (in seconds)
            t_read_total = t_read_total_v.value
            t_frameacq = t_frameacq_v.value
            t_getts = t_getts_v.value
            t_frameconv = t_frameconv_v.value

            # Measure detect_hand_pose duration
            detect_start = time.perf_counter()
            hands = detector.detect_hand_pose(frame)
            detect_end = time.perf_counter()
            detect_time = detect_end - detect_start  # seconds

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
                        counter += 1
                        print(f"Tap #{counter}")

                        # send OSC trigger
                        client.send_message('/trigger', 1)

                        # Prepare CSV row (convert seconds -> milliseconds)
                        row = [
                            time.perf_counter(),                 # record time (perf counter)
                            counter,                             # tap number
                            round(frame_age_ms, 6),              # frame age ms
                            round(t_read_total * 1000.0, 6),     # total read_frame ms
                            round(t_frameacq * 1000.0, 6),       # frame acquisition ms
                            round(t_getts * 1000.0, 6),          # get timestamps ms
                            round(t_frameconv * 1000.0, 6),      # frame conversion ms
                            round(detect_time * 1000.0, 6)       # detect_hand_pose ms
                        ]

                        # Log to CSV files
                        log_row_to_csv(row)
    except KeyboardInterrupt:
        print("CONSUMER: KeyboardInterrupt")
    finally:
        stop_event.set()
        shm0.close()
        shm1.close()
        print("CONSUMER EXITS GRACEFULLY")


if __name__ == "__main__":
    mp.set_start_method('forkserver', force=True)

    size = int(np.prod(FRAME_SHAPE) * np.dtype(FRAME_DTYPE).itemsize)
    shm0 = shared_memory.SharedMemory(create=True, size=size)
    shm1 = shared_memory.SharedMemory(create=True, size=size)

    # Shared control variables
    cur_idx = Value('i', 0)
    ts = Value('d', 0.0)
    stop_event = Event()

    # Shared timing values (seconds)
    t_read_total = Value('d', 0.0)
    t_frameacq = Value('d', 0.0)
    t_getts = Value('d', 0.0)
    t_frameconv = Value('d', 0.0)

    # Start processes
    p1 = Process(target=producer, args=(shm0.name, shm1.name,
                                        cur_idx, stop_event, ts,
                                        t_read_total, t_frameacq, t_getts, t_frameconv))
    p2 = Process(target=consumer, args=(shm0.name, shm1.name,
                                        cur_idx, stop_event, ts,
                                        t_read_total, t_frameacq, t_getts, t_frameconv))

    p1.start()
    p2.start()

    try:
        while p1.is_alive():
            p1.join(timeout=0.5)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        p1.join(timeout=1.0)
        p2.join(timeout=1.0)

        # cleanup shared memory from main process
        try:
            shm0.close()
            shm0.unlink()
        except Exception:
            pass
        try:
            shm1.close()
            shm1.unlink()
        except Exception:
            pass

        print("MAIN EXIT")
