import threading
import subprocess
import time
import re

class NPUMonitor(threading.Thread):
    def __init__(self, interval=1.0, log_file=None):
        super().__init__()
        self.interval = interval
        self.log_file = log_file
        self.running = False
        self.daemon = True  # 后台线程，不阻塞主程序退出

    def parse_memory_usage(self, output):
        lines = output.splitlines()
        
        for i, line in enumerate(lines):
            if "Memory-Usage" in line:
                if i + 3 < len(lines):
                    data_line = lines[i + 3]
                    #print(f"Parsing line: {data_line}")  # 可删
                    match = re.search(r'(\d+)\s*/\s*(\d+)', data_line)
                    if match:
                        return int(match.group(1)), int(match.group(2))
                break
        
        #print("Memory-Usage not found or pattern mismatch")  # 可删
        return None, None


    def log(self, message):
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')
        else:
            print(message)

    def run(self):
        self.running = True
        while self.running:
            try:
                result = subprocess.run(['npu-smi', 'info'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                output = result.stdout
                used, total = self.parse_memory_usage(output)
                if used is not None:
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    self.log(f'[{timestamp}] NPU Memory Usage: {used} / {total} MB')
            except Exception as e:
                self.log(f'Error: {e}')
            time.sleep(self.interval)

    def stop(self):
        self.running = False
