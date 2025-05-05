#!/usr/bin/env python
"""
Visualization script for Pensieve test results
Creates bar charts comparing all models
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import sys

def load_testing_report(filename='testing_report.json'):
    """Load the testing report JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading report: {e}")
        return None

def create_comparison_plots(report):
    """Create bar charts comparing all methods"""
    if not report or 'methods' not in report:
        print("Invalid report data")
        return
    
    methods = list(report['methods'].keys())
    bitrates = []
    rebuffers = []
    rewards = []
    
    for method in methods:
        bitrates.append(report['methods'][method].get('bitrate', 0))
        rebuffers.append(report['methods'][method].get('rebuffer', 0))
        rewards.append(report['methods'][method].get('reward', 0))
    
    # Set up the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('Comparison of Video Streaming Algorithms', fontsize=16, fontweight='bold')
    
    x = np.arange(len(methods))
    width = 0.7
    
    # Plot 1: Bitrate
    colors = []
    for method in methods:
        if 'improved' in method:
            colors.append('royalblue')
        elif 'network' in method:
            colors.append('darkgreen')
        elif 'buffer' in method:
            colors.append('darkred')
        else:
            colors.append('gray')
    
    bars1 = ax1.bar(x, bitrates, width, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Average Bitrate (kbps)', fontsize=12)
    ax1.set_title('Quality of Experience (Higher is Better)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Rebuffer Time
    bars2 = ax2.bar(x, rebuffers, width, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Average Rebuffer Time (s)', fontsize=12)
    ax2.set_title('Playback Interruptions (Lower is Better)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Reward
    bars3 = ax3.bar(x, rewards, width, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Average Reward', fontsize=12)
    ax3.set_title('Overall Performance (Higher is Better)', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        if height < 0:
            va = 'top'
        else:
            va = 'bottom'
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va=va, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a summary plot
    fig, ax = plt.figure(figsize=(10, 8))
    
    # Normalize metrics to [0, 1] for radar chart
    max_bitrate = max(bitrates) if bitrates else 1
    max_rebuffer = max(rebuffers) if rebuffers else 1
    min_reward = min(rewards) if rewards else 1
    max_reward = max(rewards) if rewards else 1
    
    # Create a simple comparison plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Normalize data for comparison
    normalized_bitrates = [(b / max_bitrate) * 100 for b in bitrates]
    normalized_rebuffers = [(1 - r / max_rebuffer) * 100 for r in rebuffers]  # Invert (lower is better)
    normalized_rewards = [((r - min_reward) / (max_reward - min_reward)) * 100 if max_reward > min_reward else 50 for r in rewards]
    
    x = np.arange(len(methods))
    width = 0.25
    
    bars1 = ax.bar(x - width, normalized_bitrates, width, label='Quality', color='lightblue', edgecolor='black')
    bars2 = ax.bar(x, normalized_rebuffers, width, label='Stability', color='lightgreen', edgecolor='black')
    bars3 = ax.bar(x + width, normalized_rewards, width, label='Overall', color='lightcoral', edgecolor='black')
    
    ax.set_ylabel('Performance Score (%)', fontsize=12)
    ax.set_title('Overall Model Performance Comparison\n(Higher is Better)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 120)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('overall_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create improvement analysis
    create_improvement_analysis(report)

def create_improvement_analysis(report):
    """Create a bar chart showing improvement over baseline"""
    # Calculate improvement over baseline
    baseline_avg = {
        'bitrate': 0,
        'rebuffer': 0,
        'reward': 0
    }
    
    baseline_methods = ['bb', 'mpc', 'dp']
    baseline_count = 0
    
    for method in baseline_methods:
        if method in report['methods']:
            for metric in baseline_avg:
                baseline_avg[metric] += report['methods'][method].get(metric, 0)
            baseline_count += 1
    
    if baseline_count > 0:
        for metric in baseline_avg:
            baseline_avg[metric] /= baseline_count
    
    # Calculate improvements for RL methods
    rl_methods = ['improved_quick', 'network_pattern', 'buffer_aware']
    improvements = {method: {} for method in rl_methods}
    
    for method in rl_methods:
        if method in report['methods']:
            improvements[method]['bitrate'] = ((report['methods'][method]['bitrate'] - baseline_avg['bitrate']) / baseline_avg['bitrate']) * 100
            improvements[method]['rebuffer'] = ((baseline_avg['rebuffer'] - report['methods'][method]['rebuffer']) / baseline_avg['rebuffer']) * 100
            improvements[method]['reward'] = ((report['methods'][method]['reward'] - baseline_avg['reward']) / baseline_avg['reward']) * 100
    
    # Create improvement plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics = ['bitrate', 'rebuffer', 'reward']
    x = np.arange(len(rl_methods))
    width = 0.25
    
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    for i, metric in enumerate(metrics):
        values = [improvements[method].get(metric, 0) for method in rl_methods]
        bars = ax.bar(x + i*width - width, values, width, label=metric.capitalize(), color=colors[i], edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va=va, fontsize=10)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Improvement over Baseline (%)', fontsize=12)
    ax.set_title('RL Model Improvements vs Baseline Methods\n(Higher is Better)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(rl_methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improvement_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function"""
    report_file = 'testing_report.json'
    if len(sys.argv) > 1:
        report_file = sys.argv[1]
    
    report = load_testing_report(report_file)
    if report:
        create_comparison_plots(report)
        print("Plots saved to current directory:")
        print("  - comparison_plots.png")
        print("  - overall_performance_comparison.png") 
        print("  - improvement_analysis.png")
    else:
        print("Failed to load testing report")

if __name__ == '__main__':
    main()
