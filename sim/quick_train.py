from __future__ import print_function
import os
import logging
import numpy as np
import multiprocessing as mp
import time
import datetime
os.environ['CUDA_VISIBLE_DEVICES']=''
import tensorflow as tf
import env
import a3c
import load_trace

# QUICK TRAINING PARAMETERS
S_INFO = 6
S_LEN = 8
A_DIM = 6
ACTOR_LR_RATE = 0.001    # INCREASED from 0.0001
CRITIC_LR_RATE = 0.01    # INCREASED from 0.001
NUM_AGENTS = 8           # REDUCED from 16
TRAIN_SEQ_LEN = 50       # REDUCED from 100
MODEL_SAVE_INTERVAL = 1  # CHANGED from 100 to save every epoch
EPOCH = 5                # QUICK training
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
SUMMARY_DIR = './quick_results'  # CHANGED to separate results
LOG_FILE = './quick_results/log'
TEST_LOG_FOLDER = './quick_test_results/'
TRAIN_TRACES = './quick_traces/'
MONITOR_FILE = './quick_results/monitor_log.txt'
NN_MODEL = None

# Create directories if they don't exist
if not os.path.exists(TRAIN_TRACES):
    os.makedirs(TRAIN_TRACES)
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)
if not os.path.exists(TEST_LOG_FOLDER):
    os.makedirs(TEST_LOG_FOLDER)

def testing(epoch, nn_model, log_file):
    # Quick testing skipped to speed up training
    pass

def log_monitor(message):
    """Logs to both console and monitor file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = "[{0}] {1}".format(timestamp, message)
    print(log_message)
    with open(MONITOR_FILE, 'a') as f:
        f.write(log_message + '\n')

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

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:
            saver.restore(sess, nn_model)
            log_monitor("Model restored.")

        epoch = 0
        start_time = time.time()

        # MAIN TRAINING LOOP with EPOCH LIMIT
        while epoch < EPOCH:
            epoch_start_time = time.time()
            
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in xrange(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])

            # record average reward and td loss change
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0 

            # assemble experiences from the agents
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

            # apply gradients
            for i in xrange(len(actor_gradient_batch)):
                actor.apply_gradients(actor_gradient_batch[i])
                critic.apply_gradients(critic_gradient_batch[i])

            # log training information
            epoch += 1
            avg_reward = total_reward  / total_agents
            avg_td_loss = total_td_loss / total_batch_len
            avg_entropy = total_entropy / total_batch_len
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time

            # ENHANCED MONITORING
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

            # Save model every epoch
            if epoch % MODEL_SAVE_INTERVAL == 0:
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt")
                log_monitor("Model saved to: {0}".format(save_path))

        log_monitor("Training completed! Total time: {0:.1f}s".format(time.time() - start_time))

def agent(agent_id, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue):
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

        # initial synchronization of the network parameters from the coordinator
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
        while True:  # experience video streaming forever

            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # -- linear reward --
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                               VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

            r_batch.append(reward)

            last_bit_rate = bit_rate

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            # compute action probability vector
            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            # log basic information
            log_file.write(str(time_stamp) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            # report experience to the coordinator
            if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:
                exp_queue.put([s_batch[1:],  # ignore the first chuck
                               a_batch[1:],  # since we don't have the
                               r_batch[1:],  # control over it
                               end_of_video,
                               {'entropy': entropy_record}])

                # synchronize the network parameters from the coordinator
                actor_net_params, critic_net_params = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del entropy_record[:]

                log_file.write('\n')  # so that in the log we know where video ends

            # store the state and action into batches
            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

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
    log_monitor("=== Starting Quick Training ===")
    log_monitor("NUM_AGENTS: {0}, EPOCH: {1}, TRAIN_SEQ_LEN: {2}".format(NUM_AGENTS, EPOCH, TRAIN_SEQ_LEN))
    
    np.random.seed(RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == A_DIM

    # create result directory
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # Create small trace set if it doesn't exist
    if not os.path.exists(TRAIN_TRACES) or len(os.listdir(TRAIN_TRACES)) < 5:
        log_monitor("Creating small training trace set...")
        import shutil
        source_traces = './cooked_test_traces/'  # Use test traces as source
        
        # Create quick_traces directory if it doesn't exist
        if not os.path.exists(TRAIN_TRACES):
            os.makedirs(TRAIN_TRACES)
            
        # Copy 5 traces for quick training
        trace_files = ['trace_0.txt', 'trace_10.txt', 'trace_20.txt', 'trace_30.txt', 'trace_40.txt']
        for i, trace_file in enumerate(trace_files):
            src = os.path.join(source_traces, trace_file)
            dst = os.path.join(TRAIN_TRACES, "trace_{0}.txt".format(i))
            if os.path.exists(src):
                shutil.copy(src, dst)
                log_monitor("Copied {0} to trace_{1}.txt".format(trace_file, i))

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in xrange(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
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

    # wait unit training is done
    coordinator.join()
    log_monitor("Training completed!")

if __name__ == '__main__':
    main()
