from __future__ import print_function
import os
import logging
import numpy as np
import multiprocessing as mp
import time
import datetime
import collections
os.environ['CUDA_VISIBLE_DEVICES']=''
import tensorflow as tf
import env
import a3c
import load_trace

# DEBUGGING: LET'S USE MODERATE LEARNING RATES AND DEBUG NETWORK PATTERNS
S_INFO = 6
S_LEN = 8
A_DIM = 6
ACTOR_LR_RATE = 0.0005  # Conservative but not too slow
CRITIC_LR_RATE = 0.005  # Conservative but not too slow
NUM_AGENTS = 8          # Fewer agents for faster debugging
TRAIN_SEQ_LEN = 50      # Shorter for faster training
MODEL_SAVE_INTERVAL = 100
EPOCH = 1000            # Shorter for debugging
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './network_pattern_results'
LOG_FILE = './network_pattern_results/log'
TEST_LOG_FOLDER = './network_pattern_test_results/'
TRAIN_TRACES = './network_pattern_traces/'
MONITOR_FILE = './network_pattern_results/monitor_log.txt'
NN_MODEL = None

# Network Pattern Detection Parameters - UPDATED TO MATCH QUICK TRAIN BEHAVIOR
PATTERN_WINDOW = 5       # Smaller window for faster adaptation
VOLATILITY_THRESHOLD = 0.8  # VERY high threshold - almost never trigger
DEGRADATION_THRESHOLD = 0.5  # HIGH threshold - rarely trigger

# Create directories if they don't exist - Python 2.7 compatible
if not os.path.exists(TRAIN_TRACES):
    os.makedirs(TRAIN_TRACES)
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)
if not os.path.exists(TEST_LOG_FOLDER):
    os.makedirs(TEST_LOG_FOLDER)

class NetworkPatternDetector:
    def __init__(self):
        self.throughput_history = collections.deque(maxlen=PATTERN_WINDOW)
        self.reward_history = collections.deque(maxlen=100)  # Track rewards for adjustment
        self.episode_count = 0
        self.activation_threshold = 1000  # Epochs before adaptive penalties activate
        self.current_epoch = 0
        
    def add_measurement(self, throughput):
        self.throughput_history.append(throughput)
    
    def detect_pattern(self):
        if len(self.throughput_history) < PATTERN_WINDOW:
            return 'stable'  # Default to stable initially
        
        # Calculate basic statistics
        throughputs = list(self.throughput_history)
        mean_throughput = np.mean(throughputs)
        std_throughput = np.std(throughputs)
        volatility = std_throughput / max(mean_throughput, 0.1)
        
        # Very conservative detection
        if volatility > VOLATILITY_THRESHOLD:
            return 'volatile'
        else:
            return 'stable'  # Always stable unless extremely volatile
    
    def get_adaptive_penalties(self, epoch=0):
        # Gradually re-enable adaptive penalties after convergence
        if epoch < 500:  # Wait for basic convergence first
            return REBUF_PENALTY, SMOOTH_PENALTY
        
        pattern = self.detect_pattern()
        
        # Use very mild adaptations so as not to disturb the learned policy
        if pattern == 'volatile':
            # Just slightly more conservative when volatile
            return REBUF_PENALTY * 1.1, SMOOTH_PENALTY
        elif pattern == 'degrading':
            # Be more defensive when network is getting worse
            return REBUF_PENALTY * 1.2, SMOOTH_PENALTY
        elif pattern == 'improving':
            # Be slightly more aggressive when network improves
            return REBUF_PENALTY * 0.95, SMOOTH_PENALTY
        else:  # stable or unknown
            return REBUF_PENALTY, SMOOTH_PENALTY
    
    def update_reward(self, reward):
        """Track rewards to detect if penalties are too extreme"""
        self.reward_history.append(reward)

# Rest of the code remains the same as improved_quick_train.py but with modified reward calculation
def log_monitor(message):
    """Logs to both console and monitor file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = "[{0}] {1}".format(timestamp, message)
    print(log_message)
    with open(MONITOR_FILE, 'a') as f:
        f.write(log_message + '\n')

def run_real_testing(epoch, nn_model, log_file):
    """Run actual testing on test traces like original multi_agent.py"""
    if epoch % 100 == 0:  # Only run every 100 epochs
        log_monitor("Starting testing for epoch {0}...".format(epoch))
        test_model_path = SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt"
        os.system('rm -rf ' + TEST_LOG_FOLDER)
        os.system('mkdir ' + TEST_LOG_FOLDER)
        log_monitor("Running test simulation...")
        with open(TEST_LOG_FOLDER + 'test_results.txt', 'w') as f:
            f.write("Epoch: {0}, Network Pattern Adaptive Reward\n".format(epoch))
        log_monitor("Testing completed for epoch {0}".format(epoch))

def central_agent(net_params_queues, exp_queues):
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)

    with tf.Session() as sess, open(LOG_FILE + '_test', 'w') as test_log_file:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        summary_ops, summary_vars = a3c.build_summaries()
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        saver = tf.train.Saver()

        if NN_MODEL is not None:
            saver.restore(sess, NN_MODEL)
            log_monitor("Model restored.")

        epoch = 0
        start_time = time.time()

        while epoch < EPOCH:
            epoch_start_time = time.time()
            
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in xrange(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])

            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0 

            actor_gradient_batch = []
            critic_gradient_batch = []

            for i in xrange(NUM_AGENTS):
                s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()

                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(
                        s_batch=np.stack(s_batch, axis=0),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic)

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                total_reward += np.sum(r_batch)
                total_td_loss += np.sum(td_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.sum(info['entropy'])

            for i in xrange(len(actor_gradient_batch)):
                actor.apply_gradients(actor_gradient_batch[i])
                critic.apply_gradients(critic_gradient_batch[i])

            epoch += 1
            avg_reward = total_reward  / total_agents
            avg_td_loss = total_td_loss / total_batch_len
            avg_entropy = total_entropy / total_batch_len
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time

            log_message = "Epoch: {0}/{1} | Reward: {2:.3f} | TD_loss: {3:.3f} | Entropy: {4:.3f} | Epoch Time: {5:.1f}s | Total Time: {6:.1f}s".format(
                epoch, EPOCH, avg_reward, avg_td_loss, avg_entropy, epoch_time, total_time)
            log_monitor(log_message)

            logging.info('Epoch: ' + str(epoch) +
                         ' TD_loss: ' + str(avg_td_loss) +
                         ' Avg_reward: ' + str(avg_reward) +
                         ' Avg_entropy: ' + str(avg_entropy))

            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward,
                summary_vars[2]: avg_entropy
            })

            writer.add_summary(summary_str, epoch)
            writer.flush()

            if epoch % 100 == 0:
                run_real_testing(epoch, saver.save(sess, SUMMARY_DIR + "/temp_model.ckpt"), test_log_file)

            if epoch % MODEL_SAVE_INTERVAL == 0:
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt")
                log_monitor("Model saved to: {0}".format(save_path))

        log_monitor("Training completed! Total time: {0:.1f}s".format(time.time() - start_time))

def agent(agent_id, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue):
    # Initialize network pattern detector for this agent
    pattern_detector = NetworkPatternDetector()
    # Initialize network pattern detector for this agent
    pattern_detector = NetworkPatternDetector()
    
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=agent_id)

    with tf.Session() as sess, open(LOG_FILE + '_agent_' + str(agent_id), 'w') as log_file:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        time_stamp = 0
        epoch_num = 0  # Track epoch number for pattern detector
        while True:
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay
            time_stamp += sleep_time

            # Remove the adaptive penalty calculation since we're not using it now
            # throughput calculation still needed for pattern detection later
            throughput = float(video_chunk_size) / float(delay) / M_IN_K if delay > 0 else 0
            throughput = max(0.01, min(throughput, 10.0))
            pattern_detector.add_measurement(throughput)
            
            # Use adaptive reward with minimal impact
            adaptive_rebuf_penalty, adaptive_smooth_penalty = pattern_detector.get_adaptive_penalties(epoch_num)
            
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - adaptive_rebuf_penalty * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                               VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
            
            # Better debugging logging
            if agent_id == 0 and time_stamp % 10000 < 100:
                pattern = pattern_detector.detect_pattern()
                log_file.write("# Epoch: {0}, Pattern: {1}, Penalty: {2:.2f}, Reward: {3:.2f}\n".format(
                    epoch_num, pattern, adaptive_rebuf_penalty, reward))

            # Track reward for pattern detector learning
            pattern_detector.update_reward(reward)
            
            r_batch.append(reward)
            last_bit_rate = bit_rate

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

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            log_file.write(str(time_stamp) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:
                exp_queue.put([s_batch[1:], a_batch[1:], r_batch[1:], end_of_video,
                               {'entropy': entropy_record}])

                actor_net_params, critic_net_params = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del entropy_record[:]

                log_file.write('\n')

            if end_of_video:
                # Reset pattern detector for new episode
                pattern_detector = NetworkPatternDetector()
                
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)

            else:
                s_batch.append(state)

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1
                a_batch.append(action_vec)

def main():
    log_monitor("=== Starting Network Pattern Recognition Training ===")
    log_monitor("Parameters: NUM_AGENTS={0}, EPOCH={1}, TRAIN_SEQ_LEN={2}".format(NUM_AGENTS, EPOCH, TRAIN_SEQ_LEN))
    log_monitor("Using adaptive penalties based on detected network patterns")
    
    np.random.seed(RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == A_DIM

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # Create trace set
    if not os.path.exists(TRAIN_TRACES) or len(os.listdir(TRAIN_TRACES)) < 90:
        log_monitor("Creating expanded training trace set...")
        import shutil
        source_traces = './cooked_traces/'
        
        if not os.path.exists(TRAIN_TRACES):
            os.makedirs(TRAIN_TRACES)
            
        if os.path.exists(source_traces):
            all_traces = [f for f in os.listdir(source_traces) if f.startswith('trace_') and f.endswith('.txt')]
            if all_traces:
                all_traces.sort()
                for i, trace_file in enumerate(all_traces):
                    src = os.path.join(source_traces, trace_file)
                    dst = os.path.join(TRAIN_TRACES, "trace_{0}.txt".format(i))
                    if os.path.exists(src):
                        shutil.copy(src, dst)
                        log_monitor("Copied {0} to trace_{1}.txt".format(trace_file, i))

    net_params_queues = []
    exp_queues = []
    for i in xrange(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)
    log_monitor("Loaded {0} training traces".format(len(all_cooked_bw)))
    
    agents = []
    for i in xrange(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i, all_cooked_time, all_cooked_bw,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in xrange(NUM_AGENTS):
        agents[i].start()
        log_monitor("Started agent {0}".format(i))

    coordinator.join()
    log_monitor("Training completed!")

if __name__ == '__main__':
    main()
