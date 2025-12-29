import os
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent

# === CONFIGURATION ===
BASE_DIR = os.environ.get("BSM_WORKING_DIR", os.path.join(os.getcwd(), "data", "working"))
OUTPUT_BASE = r"C:\Users\asaha96\Desktop\AritraDirectory\OUTPUT_BASE"
PARALLEL_SCRIPT = r"C:\Users\asaha96\Desktop\AritraDirectory\GOPI_SCRIPTS\parallel_core_pcap_to_csv_gopi.py"
PYTHON_EXEC = r"C:\Program Files\Python38\python.exe"

class PcapHandler(FileSystemEventHandler):
    def on_created(self, event):
        if isinstance(event, FileCreatedEvent) and event.src_path.lower().endswith(('.pcap', '.pcapng')):
            self.process(event.src_path)

    def on_modified(self, event):
        if isinstance(event, FileModifiedEvent) and event.src_path.lower().endswith(('.pcap', '.pcapng')):
            self.process(event.src_path)

    def process(self, pcap_path: str):
        print(f"[INFO] Detected: {pcap_path}")
        try:
            rel_path = os.path.relpath(pcap_path, BASE_DIR)
        except ValueError:
            print(f"[ERROR] File outside BASE_DIR: {pcap_path}")
            return

        output_csv = os.path.splitext(rel_path)[0] + ".csv"
        output_path = os.path.join(OUTPUT_BASE, output_csv)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cmd = [PYTHON_EXEC, PARALLEL_SCRIPT, "--input", pcap_path, "--output", output_path]
        print(f"[INFO] Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"[SUCCESS] Saved: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Processing failed: {e}")

def run_monitor():
    print(f"[INFO] Watching: {BASE_DIR}")
    handler = PcapHandler()
    observer = Observer()
    observer.schedule(handler, BASE_DIR, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[INFO] Stopping monitor.")
        observer.stop()
    observer.join()

if __name__ == "__main__":
    # 1) Batch‐mode pass to cover any PCAPs that arrived while the monitor was down
    #subprocess.run(
        #[PYTHON_EXEC, PARALLEL_SCRIPT],
        #check=True
    #)

    # 2) Then kick off the real-time directory watcher
    run_monitor()
