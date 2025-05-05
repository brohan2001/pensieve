import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import tensorflow as tf
import fixed_env as env
import a3c
import load_trace
import matplotlib.pyplot as plt
import time
import datetime

# QUICK TEST PARAMETERS
S_INFO = 6
S_LEN = 8
A_DIM = 6
ACTOR_LR_RATE = 0.001
CRITIC_LR_RATE = 0.01
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './quick_test_results'
LOG_FILE = './quick_test_results/log_sim_rl'
QOE_LOG_FILE = './quick_test_results/qoe_log.txt'
MONITOR_FILE = './quick_test_results/monitor_log.txt'
STATS_FILE = './quick_test_results/stats_summary.txt'

# Use the quick-trained model
NN_MODEL = '../sim/quick_results/nn_model_ep_5.ckpt'

# Quick trace set
QUICK_TRACE_PATH = './quick_test_traces/'

# Create directories
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)
if not os.path.exists(QUICK_TRACE_PATH):
    os.makedirs(QUICK_TRACE_PATH)

def log_monitor(message):
    """Logs to both console and monitor file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = "[{0}] {1}".format(timestamp, message)

    print(log_message)
    with open(MONITOR_FILE, 'a') as f:
        f.write(log_message + '\n')

def setup_quick_traces():
    """Create a small test set if it doesn't exist"""
    source_path = '../sim/cooked_test_traces/'

    # Use actual test trace filenames
    trace_files = [
        'bus.ljansbakken-oslo-report.2010-09-29_0852CEST.log_300.txt',
        'ferry.nesoddtangen-oslo-report.2010-09-20_1542CEST.log_0.txt',
        'metro.kalbakken-jernbanetorget-report.2010-09-21_0742CEST.log_180.txt',
        'train.oslo-vestby-report.2011-02-11_1618CET.log_60.txt',
        'tram.jernbanetorget-ljabru-report.2010-12-16_1100CET.log_0.txt'
    ]

    for trace_file in trace_files:
        src = os.path.join(source_path, trace_file)
        dst = os.path.join(QUICK_TRACE_PATH, trace_file)
        if os.path.exists(src) and not os.path.exists(dst):
            import shutil
            shutil.copy(src, dst)
            log_monitor("Copied {0} to quick_test_traces".format(trace_file))

def calculate_qoe_metrics(log_file):
    """Calculate QoE metrics from log file"""
    bitrates = []
    rebuffers = []
    rewards = []
    smoothness_penalties = []

    try:
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 7:
                        bitrate = float(parts[1])
                        rebuffer = float(parts[3])
                        reward = float(parts[6])

                        bitrates.append(bitrate)
                        rebuffers.append(rebuffer)
                        rewards.append(reward)

                        if len(bitrates) > 1:
                            smoothness = abs(bitrates[-1] - bitrates[-2])
                            smoothness_penalties.append(smoothness)
    except Exception as e:
        log_monitor("Error reading log file log_file {0}: {1}".format(log_file, e))
        return {}

    if not bitrates:
        return {}

    metrics = {
        'avg_bitrate': np.mean(bitrates),
        'total_rebuffer': np.sum(rebuffers),
        'avg_reward': np.mean(rewards),
        'total_reward': np.sum(rewards),
        'avg_smoothness': np.mean(smoothness_penalties) if smoothness_penalties else 0,
        'bitrate_std': np.std(bitrates),
        'rebuffer_events': len([r for r in rebuffers if r > 0])
    }

    return metrics

def main():
    log_monitor("=== Starting Quick Testing ===")
    log_monitor("Testing model: {0}".format(NN_MODEL))

    # Setup quick traces
    setup_quick_traces()

    np.random.seed(RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == A_DIM

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # Load quick test traces
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(QUICK_TRACE_PATH)
    log_monitor("Loaded {0} test traces".format(len(all_file_names)))

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    # Open QoE log file
    qoe_log = open(QOE_LOG_FILE, 'w')
    qoe_log.write("trace\tavg_bitrate\ttotal_rebuffer\tavg_reward\ttotal_reward\tavg_smoothness\n")

    # Dictionary to store all metrics
    all_metrics = {}

    with tf.Session() as sess:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)

        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:
            saver.restore(sess, nn_model)
            log_monitor("Model restored.")

        video_count = 0
        start_time = time.time()

        while video_count < len(all_file_names):
            trace_start_time = time.time()
            current_trace = all_file_names[net_env.trace_idx]
            log_path = LOG_FILE + '_' + current_trace
            log_file = open(log_path, 'wb')

            time_stamp = 0
            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch = [np.zeros((S_INFO, S_LEN))]
            a_batch = [action_vec]
            r_batch = []
            entropy_record = []

            while True:  # simulate video streaming
                delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain = \
                    net_env.get_video_chunk(bit_rate)

                time_stamp += delay
                time_stamp += sleep_time

                # reward calculation
                reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                         - REBUF_PENALTY * rebuf \
                         - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                                   VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

                r_batch.append(reward)
                last_bit_rate = bit_rate

                # log information
                log_file.write(str(time_stamp / M_IN_K) + '\t' +
                               str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                               str(buffer_size) + '\t' +
                               str(rebuf) + '\t' +
                               str(video_chunk_size) + '\t' +
                               str(delay) + '\t' +
                               str(reward) + '\n')
                log_file.flush()

                # state update
                if len(s_batch) == 0:
                    state = [np.zeros((S_INFO, S_LEN))]
                else:
                    state = np.array(s_batch[-1], copy=True)

                state = np.roll(state, -1, axis=1)

                state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
                state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
                state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K
                state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
                state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
                state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

                action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
                action_cumsum = np.cumsum(action_prob)
                bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()

                s_batch.append(state)
                entropy_record.append(a3c.compute_entropy(action_prob[0]))

                if end_of_video:
                    log_file.write('\n')
                    log_file.close()

                    # Calculate QoE metrics for this trace
                    metrics = calculate_qoe_metrics(log_path)
                    all_metrics[current_trace] = metrics

                    if metrics:
                        qoe_log.write("{0}\t{1:.2f}\t{2:.1f}\t{3:.3f}\t{4:.1f}\t{5:.2f}\n".format(
                            current_trace,
                            metrics['avg_bitrate'],
                            metrics['total_rebuffer'],
                            metrics['avg_reward'],
                            metrics['total_reward'],
                            metrics['avg_smoothness']
                        ))
                        qoe_log.flush()

                        trace_time = time.time() - trace_start_time
                        log_monitor("Trace {0}/{1}: {2}".format(video_count + 1, len(all_file_names), current_trace))
                        log_monitor("  Avg Bitrate: {0:.1f} kbps".format(metrics['avg_bitrate']))
                        log_monitor("  Total Rebuffer: {0:.1f}s".format(metrics['total_rebuffer']))
                        log_monitor("  Avg Reward: {0:.3f}".format(metrics['avg_reward']))
                        log_monitor("  Time: {0:.1f}s".format(trace_time))

                    last_bit_rate = DEFAULT_QUALITY
                    bit_rate = DEFAULT_QUALITY

                    del s_batch[:]
                    del a_batch[:]
                    del r_batch[:]

                    action_vec = np.zeros(A_DIM)
                    action_vec[bit_rate] = 1

                    s_batch.append(np.zeros((S_INFO, S_LEN)))
                    a_batch.append(action_vec)
                    entropy_record = []

                    video_count += 1
                    break

    qoe_log.close()

    total_time = time.time() - start_time
    log_monitor("Testing completed in {0:.1f}s".format(total_time))

    with open(STATS_FILE, 'w') as f:
        f.write("=== Quick Test Summary ===\n")
        f.write("Model: {0}\n".format(NN_MODEL))
        f.write("Total testing time: {0:.1f}s\n".format(total_time))
        f.write("Number of traces: {0}\n\n".format(len(all_file_names)))

        if all_metrics:
            all_bitrates = []
            all_rebuffers = []
            all_rewards = []

            for trace, metrics in all_metrics.items():
                if metrics:
                    all_bitrates.append(metrics['avg_bitrate'])
                    all_rebuffers.append(metrics['total_rebuffer'])
                    all_rewards.append(metrics['avg_reward'])

            if all_bitrates:
                f.write("=== Overall Statistics ===\n")
                f.write("Average bitrate: {0:.2f} +/- {1:.2f} kbps\n".format(np.mean(all_bitrates), np.std(all_bitrates)))
                f.write("Average total rebuffer: {0:.2f} +/- {1:.2f}s\n".format(np.mean(all_rebuffers), np.std(all_rebuffers)))
                f.write("Average reward: {0:.3f} +/- {1:.3f}\n".format(np.mean(all_rewards), np.std(all_rewards)))

    log_monitor("Summary statistics written to stats_summary.txt")

if __name__ == '__main__':
    main()
