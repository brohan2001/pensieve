import os

# Read original mpc.py
with open('mpc.py', 'r') as f:
    content = f.read()

# Find and replace the problematic line
lines = content.split('\n')
fixed_lines = []

for i, line in enumerate(lines):
    if 'net_env.get_video_chunk(bit_rate)' in line and i > 0:
        # Look for the previous assignment line
        prev_line = lines[i-1] if i > 0 else ''
        if 'next_video_chunk_sizes, end_of_video, video_chunk_remain' in prev_line:
            # Replace with fixed version
            fixed_lines.append('        result = net_env.get_video_chunk(bit_rate)')
            fixed_lines.append('        if len(result) == 7:')
            fixed_lines.append('            delay, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = result')
            fixed_lines.append('            sleep_time = 0')
            fixed_lines.append('        else:')
            fixed_lines.append('            delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = result')
        else:
            fixed_lines.append(line)
    else:
        fixed_lines.append(line)

# Write fixed version
with open('mpc_fixed.py', 'w') as f:
    f.write('\n'.join(fixed_lines))
