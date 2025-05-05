import glob
import numpy as np

log_files = glob.glob('results/log_*')
print("Found {} log files".format(len(log_files)))

bitrates = []
rebuffers = []
rewards = []

for log_file in log_files:
    with open(log_file, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            values = line.strip().split()
            if len(values) >= 7:
                try:
                    bitrate = float(values[1])
                    rebuffer = float(values[3])
                    reward = float(values[6])
                    if reward != 0:  # Skip empty chunks
                        bitrates.append(bitrate)
                        rebuffers.append(rebuffer)
                        rewards.append(reward)
                except:
                    continue

if bitrates:
    print("Average bitrate: {:.2f}".format(np.mean(bitrates)))
    print("Average rebuffer: {:.2f}".format(np.mean(rebuffers)))
    print("Average reward: {:.2f}".format(np.mean(rewards)))
else:
    print("No valid data found")
