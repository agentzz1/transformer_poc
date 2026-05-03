import time
import os
import re

log_file = "sim.log"
last_line = ""

print("Monitoring sim.log activity...")
while True:
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            lines = f.readlines()
            if lines:
                new_last_line = lines[-1].strip()
                if new_last_line != last_line:
                    print(f"Update detected: {new_last_line}")
                    last_line = new_last_line
                else:
                    # Check if simulation is still running
                    pass
    time.sleep(2)
