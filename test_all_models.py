#!/usr/bin/env python
"""
Automated testing script for all Pensieve models
Runs tests for improved_quick, network_pattern, buffer_aware models plus baselines
"""

import os
import shutil
import subprocess
import json
import time
from datetime import datetime
import glob

# Model configurations
models = {
    'improved_quick': '../sim/improved_quick_results/nn_model_ep_1000.ckpt',
    'network_pattern': '../sim/network_pattern_results/nn_model_ep_1000.ckpt',
    'buffer_aware': '../sim/buffer_aware_results/nn_model_ep_1000.ckpt'
}

baseline_methods = ['bb', 'mpc', 'dp']

def run_command(cmd, cwd='.'):
    try:
        output = subprocess.check_output(cmd, shell=True, cwd=cwd, stderr=subprocess.STDOUT)
        return True, output.decode('utf-8')
    except subprocess.CalledProcessError as e:
        return False, e.output.decode('utf-8')

def copy_model(source_path, dest_path):
    related_files = [
        source_path,
        source_path + '.data-00000-of-00001',
        source_path + '.index',
        source_path + '.meta'
    ]
    for file_path in related_files:
        if os.path.exists(file_path):
            shutil.copy(file_path, os.path.join('models', os.path.basename(file_path)))
            print("Copied {0}".format(file_path))

def modify_rl_no_training(model_name):
    script_path = 'rl_no_training.py'
    model_line = "NN_MODEL = './models/nn_model_ep_1000.ckpt'\n"
    with open(script_path, 'r') as f:
        lines = f.readlines()
    with open(script_path, 'w') as f:
        for line in lines:
            if line.startswith('NN_MODEL ='):
                f.write(model_line)
            else:
                f.write(line)

def save_results(method_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if os.path.exists('./test_results'):
        shutil.move('./test_results', './test_results_{0}_{1}'.format(method_name, timestamp))
    for ext in ['txt', 'log']:
        src_pattern = 'summary*' if ext == 'txt' else 'log*'
        for file_path in glob.glob(src_pattern):
            base_name = os.path.basename(file_path)
            new_name = "{0}_{1}".format(method_name, base_name)
            shutil.move(file_path, new_name)
            print("Saved {0}".format(new_name))

def test_rl_model(model_name, model_path):
    print("\n{0} Testing {1} {0}".format('='*20, model_name))
    if not os.path.exists('models'):
        os.makedirs('models')
    print("Copying {0} checkpoint...".format(model_name))
    copy_model(model_path, 'models')
    print("Modifying rl_no_training.py...")
    modify_rl_no_training(model_name)
    print("Running get_video_sizes.py...")
    success, output = run_command('python get_video_sizes.py')
    if not success:
        print("Error running get_video_sizes.py: {0}".format(output))
        return False
    print("Running rl_no_training.py...")
    success, output = run_command('python rl_no_training.py')
    if not success:
        print("Error running rl_no_training.py: {0}".format(output))
        return False
    print("Saving results...")
    save_results(model_name)
    return True

def test_baseline(method_name):
    print("\n{0} Testing {1} {0}".format('='*20, method_name))
    print("Running get_video_sizes.py...")
    success, output = run_command('python get_video_sizes.py')
    if not success:
        print("Error running get_video_sizes.py: {0}".format(output))
        return False
    print("Running {0}.py...".format(method_name))
    if method_name == 'dp':
        success, output = run_command('./dp')
    else:
        success, output = run_command('python {0}.py'.format(method_name))
    if not success:
        print("Error running {0}: {1}".format(method_name, output))
        return False
    print("Saving results...")
    save_results(method_name)
    return True

def extract_metrics(summary_file):
    if not os.path.exists(summary_file):
        return None
    with open(summary_file, 'r') as f:
        lines = f.readlines()
    metrics = {}
    for line in lines:
        if 'average bitrate:' in line:
            metrics['bitrate'] = float(line.split('average bitrate:')[1].split('+/-')[0].strip())
        elif 'average total rebuffer:' in line:
            metrics['rebuffer'] = float(line.split('average total rebuffer:')[1].split('+/-')[0].strip())
        elif 'average reward:' in line:
            metrics['reward'] = float(line.split('average reward:')[1].split('+/-')[0].strip())
    return metrics

def generate_comparison_report():
    report = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'methods': {}
    }
    all_methods = list(models.keys()) + baseline_methods
    for method in all_methods:
        summary_files = [f for f in os.listdir('.') if f.startswith("{0}_summary".format(method)) and f.endswith('.txt')]
        if summary_files:
            latest_summary = max(summary_files, key=lambda f: os.path.getctime(f))
            metrics = extract_metrics(latest_summary)
            if metrics:
                report['methods'][method] = metrics
    with open('testing_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    print("\n" + "="*60)
    print("Testing Report Summary")
    print("="*60)
    print("Generated at: {0}".format(report['timestamp']))
    print("-"*60)
    print("{0:<20} {1:<15} {2:<15} {3}".format('Method', 'Bitrate', 'Rebuffer', 'Reward'))
    print("-"*60)
    for method, metrics in report['methods'].items():
        print("{0:<20} {1:<15.2f} {2:<15.2f} {3:.2f}".format(
            method,
            metrics.get('bitrate', 0),
            metrics.get('rebuffer', 0),
            metrics.get('reward', 0)))
    print("="*60)
    if report['methods']:
        best_bitrate = max(report['methods'].items(), key=lambda x: x[1].get('bitrate', 0))
        best_rebuffer = min(report['methods'].items(), key=lambda x: x[1].get('rebuffer', float('inf')))
        best_reward = max(report['methods'].items(), key=lambda x: x[1].get('reward', float('-inf')))
        print("\nBest Performing Models:")
        print("Highest Bitrate: {0} ({1:.2f})".format(best_bitrate[0], best_bitrate[1]['bitrate']))
        print("Lowest Rebuffer: {0} ({1:.2f})".format(best_rebuffer[0], best_rebuffer[1]['rebuffer']))
        print("Highest Reward: {0} ({1:.2f})".format(best_reward[0], best_reward[1]['reward']))
        print("="*60)

def main():
    if not os.path.exists('test'):
        print("Please run this script from the project root directory")
        return
    os.chdir('test')
    for model_name, model_path in models.items():
        test_rl_model(model_name, model_path)
        time.sleep(2)
    for method in baseline_methods:
        test_baseline(method)
        time.sleep(2)
    generate_comparison_report()
    print("\nAll testing completed! Check 'testing_report.json' for detailed results.")

if __name__ == '__main__':
    main()
