#!/usr/bin/env python
"""
Process test results and create summary files
"""

import os
import glob
import numpy as np
import json
from datetime import datetime

def process_log_files(log_dir, method_name):
    """Process log files and calculate statistics"""
    log_files = glob.glob(os.path.join(log_dir, 'log_sim_*'))
    
    bitrates = []
    rebuffers = []
    rewards = []
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            trace_bitrate = 0
            trace_rebuffer = 0
            trace_reward = 0
            chunk_count = 0
            
            for line in f:
                # Skip headers
                if line.startswith("#"):
                    continue
                    
                values = line.strip().split()
                if len(values) < 4:
                    continue
                    
                try:
                    bitrate = float(values[1])
                    rebuffer = float(values[3])
                    reward = float(values[6])
                    
                    trace_bitrate += bitrate
                    trace_rebuffer += rebuffer
                    trace_reward += reward
                    chunk_count += 1
                except (IndexError, ValueError):
                    continue
            
            if chunk_count > 0:
                bitrates.append(trace_bitrate / chunk_count)
                rebuffers.append(trace_rebuffer)
                rewards.append(trace_reward)
    
    if not bitrates:
        return None
    
    # Calculate statistics
    return {
        'avg_bitrate': np.mean(bitrates),
        'std_bitrate': np.std(bitrates),
        'avg_rebuffer': np.mean(rebuffers),
        'std_rebuffer': np.std(rebuffers),
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'num_traces': len(bitrates)
    }

def create_summary_file(stats, method_name):
    """Create a summary file in the expected format"""
    if stats is None:
        return False
    
    filename = 'summary_{0}.txt'.format(method_name)
    with open(filename, 'w') as f:
        f.write('average bitrate: {0:.2f} +/- {1:.2f}\n'.format(stats['avg_bitrate'], stats['std_bitrate']))
        f.write('average total rebuffer: {0:.2f} +/- {1:.2f}\n'.format(stats['avg_rebuffer'], stats['std_rebuffer']))
        f.write('average reward: {0:.2f} +/- {1:.2f}\n'.format(stats['avg_reward'], stats['std_reward']))
        f.write('total testing traces: {0}\n'.format(stats['num_traces']))
    
    print("Created {0}".format(filename))
    return True

def create_testing_report(results):
    """Create a comprehensive testing report"""
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'methods': {}
    }
    
    for method, stats in results.items():
        if stats is not None:
            report['methods'][method] = {
                'bitrate': stats['avg_bitrate'],
                'rebuffer': stats['avg_rebuffer'],
                'reward': stats['avg_reward']
            }
    
    with open('testing_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    print("Updated testing_report.json")

def main():
    # Process existing results directories
    results = {}
    
    # Look for results directories from previous tests
    test_dirs = [
        ('test_results_improved_quick', 'improved_quick'),
        ('test_results_network_pattern', 'network_pattern'),  
        ('test_results_buffer_aware', 'buffer_aware'),
        ('test_results_bb', 'bb'),
        ('test_results_mpc', 'mpc'),
        ('test_results_dp', 'dp')
    ]
    
    for test_dir, method_name in test_dirs:
        if os.path.exists(test_dir):
            stats = process_log_files(test_dir, method_name)
            results[method_name] = stats
            create_summary_file(stats, method_name)
    
    # Also process the current results directory
    if os.path.exists('results'):
        stats = process_log_files('results', 'current_test')
        results['current_test'] = stats
        create_summary_file(stats, 'current_test')
    
    # Create comprehensive report
    create_testing_report(results)
    
    # Print summary
    print("\n" + "="*50)
    print("Test Results Summary")
    print("="*50)
    print("{0:<20} {1:<15} {2:<15} {3}".format('Method', 'Bitrate', 'Rebuffer', 'Reward'))
    print("-"*70)
    
    for method, stats in results.items():
        if stats is not None:
            print("{0:<20} {1:<15.2f} {2:<15.2f} {3:.2f}".format(
                method,
                stats['avg_bitrate'],
                stats['avg_rebuffer'],
                stats['avg_reward']
            ))
    print("="*50)

if __name__ == '__main__':
    main()
