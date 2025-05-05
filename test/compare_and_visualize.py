#!/usr/bin/env python
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np

# Methods to compare (excluding dp)
methods = ['improved_quick', 'network_pattern', 'buffer_aware', 'bb', 'mpc']
results = {}

# Read summary files
for method in methods:
    filename = 'summary_{}.txt'.format(method)
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            content = f.read()
            lines = content.strip().split('\n')
            
            try:
                # Extract values more carefully
                for line in lines:
                    if 'Average bitrate:' in line:
                        bitrate = float(line.split(': ')[1])
                    elif 'Average rebuffer:' in line:
                        rebuffer = float(line.split(': ')[1])
                    elif 'Average reward:' in line:
                        reward = float(line.split(': ')[1])
                
                results[method] = {
                    'bitrate': bitrate,
                    'rebuffer': rebuffer,
                    'reward': reward
                }
            except Exception as e:
                print("Error parsing {}: {}".format(method, e))

# Print text comparison table
print("Performance Comparison (All Methods)")
print("="*70)
print("{:<20} {:>15} {:>15} {:>15}".format("Method", "Bitrate (kbps)", "Rebuffer (s)", "Reward"))
print("-"*70)

for method in methods:
    if method in results:
        r = results[method]
        print("{:<20} {:>15.2f} {:>15.2f} {:>15.2f}".format(
            method, r['bitrate'], r['rebuffer'], r['reward']))
    else:
        print("{:<20} {:>15} {:>15} {:>15}".format(method, "Failed", "Failed", "Failed"))

print("="*70)

# Find best in each category
if results:
    best_bitrate = max(results.items(), key=lambda x: x[1]['bitrate'])
    best_rebuffer = min(results.items(), key=lambda x: x[1]['rebuffer'])
    best_reward = max(results.items(), key=lambda x: x[1]['reward'])
    
    print("\nBest Performance:")
    print("Highest Bitrate: {} ({:.2f} kbps)".format(best_bitrate[0], best_bitrate[1]['bitrate']))
    print("Lowest Rebuffer: {} ({:.2f} s)".format(best_rebuffer[0], best_rebuffer[1]['rebuffer']))
    print("Highest Reward: {} ({:.2f})".format(best_reward[0], best_reward[1]['reward']))
    print("="*70)

# Create visualizations
if results:
    # Extract data for plotting
    method_names = []
    bitrates = []
    rebuffers = []
    rewards = []
    
    for method in methods:
        if method in results:
            method_names.append(method)
            bitrates.append(results[method]['bitrate'])
            rebuffers.append(results[method]['rebuffer'])
            rewards.append(results[method]['reward'])
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('Pensieve Model Performance Comparison', fontsize=16, fontweight='bold')
    
    x = np.arange(len(method_names))
    width = 0.6
    
    # Plot 1: Bitrate
    bars1 = ax1.bar(x, bitrates, width, color=['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    ax1.set_ylabel('Bitrate (kbps)', fontsize=12)
    ax1.set_title('Average Video Quality', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                '{:.1f}'.format(height), ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Rebuffer Time
    bars2 = ax2.bar(x, rebuffers, width, color=['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    ax2.set_ylabel('Rebuffer Time (s)', fontsize=12)
    ax2.set_title('Playback Interruptions (Lower is Better)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                '{:.3f}'.format(height), ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Reward
    bars3 = ax3.bar(x, rewards, width, color=['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    ax3.set_ylabel('Reward', fontsize=12)
    ax3.set_title('Overall Performance Score', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(method_names, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        if height < 0:
            va = 'top'
        else:
            va = 'bottom'
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                '{:.2f}'.format(height), ha='center', va=va, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved performance comparison plot to 'performance_comparison.png'")
    
    # Create a normalized performance comparison
    fig, ax = plt.figure(figsize=(12, 8))
    
    # Normalize data to [0,1] scale for each metric
    max_bitrate = max(bitrates) if bitrates else 1
    min_rebuffer = min(rebuffers) if rebuffers else 1
    max_rebuffer = max(rebuffers) if rebuffers else 1
    min_reward = min(rewards) if rewards else 1
    max_reward = max(rewards) if rewards else 1
    
    normalized_bitrates = [(b / max_bitrate) * 100 for b in bitrates]
    normalized_rebuffers = [((max_rebuffer - r) / max_rebuffer) * 100 for r in rebuffers]  # Lower is better
    normalized_rewards = [((r - min_reward) / (max_reward - min_reward)) * 100 if max_reward > min_reward else 50 for r in rewards]
    
    x = np.arange(len(method_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, normalized_bitrates, width, label='Quality', color='lightblue', edgecolor='black')
    bars2 = ax.bar(x, normalized_rebuffers, width, label='Stability', color='lightgreen', edgecolor='black')
    bars3 = ax.bar(x + width, normalized_rewards, width, label='Overall', color='lightcoral', edgecolor='black')
    
    ax.set_ylabel('Performance Score (%)', fontsize=12)
    ax.set_title('Normalized Performance Comparison\n(Higher is Better)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 120)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    '{:.0f}%'.format(height), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('normalized_performance.png', dpi=300, bbox_inches='tight')
    print("Saved normalized performance plot to 'normalized_performance.png'")
    
    # Create radar chart comparing RL models vs baseline
    categories = ['Quality', 'Stability', 'Overall']
    
    # Calculate average performance of RL models
    rl_methods = ['improved_quick', 'network_pattern', 'buffer_aware']
    rl_results = []
    baseline_results = []
    
    for metric in ['bitrate', 'rebuffer', 'reward']:
        rl_values = []
        baseline_values = []
        
        for method in methods:
            if method in results:
                if method in rl_methods:
                    rl_values.append(results[method][metric])
                else:
                    baseline_values.append(results[method][metric])
        
        if rl_values:
            rl_results.append(np.mean(rl_values))
        if baseline_values:
            baseline_results.append(np.mean(baseline_values))
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.figure(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Plot data
    if rl_results and baseline_results:
        # Normalize data
        rl_norm = []
        baseline_norm = []
        for i in range(len(categories)):
            if i == 1:  # Rebuffer - invert because lower is better
                max_val = max(rl_results[i], baseline_results[i])
                rl_norm.append((max_val - rl_results[i]) / max_val)
                baseline_norm.append((max_val - baseline_results[i]) / max_val)
            else:
                max_val = max(rl_results[i], baseline_results[i])
                rl_norm.append(rl_results[i] / max_val if max_val > 0 else 0)
                baseline_norm.append(baseline_results[i] / max_val if max_val > 0 else 0)
        
        rl_norm += rl_norm[:1]
        baseline_norm += baseline_norm[:1]
        
        ax.plot(angles, rl_norm, 'o-', linewidth=2, label='RL Models (avg)', color='red')
        ax.fill(angles, rl_norm, alpha=0.25, color='red')
        ax.plot(angles, baseline_norm, 'o-', linewidth=2, label='Baseline (avg)', color='blue')
        ax.fill(angles, baseline_norm, alpha=0.25, color='blue')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.set_title('RL Models vs Baseline Comparison', fontsize=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        plt.tight_layout()
        plt.savefig('rl_vs_baseline_radar.png', dpi=300, bbox_inches='tight')
        print("Saved radar chart to 'rl_vs_baseline_radar.png'")
    
    print("\nAll visualization files saved to current directory.")
else:
    print("\nNo valid results to visualize.")
