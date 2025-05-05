#!/usr/bin/env python

import os
import glob
import json

print("Checking test results...")
print("========================")

# Check for results directories
for method in ['buffer_aware', 'network_pattern', 'improved_quick', 'bb', 'mpc']:
    results_dir = 'test_results_{0}'.format(method)
    
    print("\n{0}:".format(method))
    
    if os.path.exists(results_dir):
        print("  Directory exists: Yes")
        
        # Check for log files
        log_files = glob.glob(os.path.join(results_dir, 'log_*'))
        print("  Log files found: {0}".format(len(log_files)))
        
        # Check if any log files contain data
        if log_files:
            with open(log_files[0], 'r') as f:
                lines = f.readlines()
            print("  Lines in first log: {0}".format(len(lines)))
            
            # Check content of first log file
            if len(lines) > 1:
                print("  First data line: {0}".format(lines[1].strip()[:100]))
        else:
            print("  No log files found")
    else:
        print("  Directory exists: No")

# Check for summary files
print("\nSummary files:")
summary_files = glob.glob('summary_*.txt')
if summary_files:
    for sf in summary_files:
        print("  Found: {0}".format(sf))
else:
    print("  No summary files found")

# Check testing report
if os.path.exists('testing_report.json'):
    print("\nTesting report exists:")
    with open('testing_report.json', 'r') as f:
        report = json.load(f)
        print("  Methods in report: {0}".format(list(report.get('methods', {}).keys())))
else:
    print("\nTesting report does not exist")
