import os
import argparse
import pandas as pd
from collections import defaultdict
import re

def extract_success_rate(filename):
    """Extract success rate from filename like '12_SR0.0' or '11_SR0'"""
    match = re.search(r'_SR(\d+(?:\.\d+)?)', filename)
    if match:
        return float(match.group(1))
    return None

def analyze_level_folder(level_path):
    """Analyze a single level folder (test_l2, test_l3, test_l4)"""
    videos_path = os.path.join(level_path, 'videos')
    if not os.path.exists(videos_path):
        print(f"Warning: videos path not found: {videos_path}")
        return {}
    
    task_stats = {}
    for task_name in os.listdir(videos_path):
        task_path = os.path.join(videos_path, task_name)
        if not os.path.isdir(task_path):
            continue
            
        # Count successes
        total_episodes = 0
        successes = 0
        for video_file in os.listdir(task_path):
            if video_file.endswith('_SR0.0') or video_file.endswith('_SR0'):
                total_episodes += 1
            else:
                total_episodes += 1
                successes += 1
        
        if total_episodes > 0:
            success_rate = (successes / total_episodes) * 100
            task_stats[task_name] = {
                'successes': successes,
                'total': total_episodes,
                'success_rate': success_rate
            }
    
    return task_stats

def main():
    results_dir = "/home/ksaha/Research/ModelBasedPlanning/PriorWork/robot-3dlotus/data/experiments/gembench/3dlotus/v1/preds/seed100"
    output_file = os.path.join(results_dir, "benchmark_results.csv")

    # Dictionary to store all results
    all_results = {}
    
    # Analyze each level
    for level in ['test_l2', 'test_l3', 'test_l4']:
        level_path = os.path.join(results_dir, level)
        if os.path.exists(level_path):
            all_results[level] = analyze_level_folder(level_path)
        else:
            raise ValueError(f"Level path not found: {level_path}")

    # Create DataFrame
    rows = []
    for level, tasks in all_results.items():
        level_num = level.split('_')[1]  # 'test_l2' -> 'l2'
        for task_name, stats in tasks.items():
            rows.append({
                'level': level_num,
                'task': task_name,
                'successes': stats['successes'],
                'total_episodes': stats['total'],
                'success_rate': stats['success_rate']
            })
    
    df = pd.DataFrame(rows)
    
    # Calculate level-wise averages
    level_stats = df.groupby('level')['success_rate'].agg(['mean', 'std']).round(2)
    level_stats.columns = ['Average Success Rate', 'Std Dev']
    
    # Save detailed results
    df.to_csv(output_file, index=False)
    
    # Print summary
    print("\nLevel-wise Statistics:")
    print(level_stats)
    print(f"\nDetailed results saved to: {output_file}")

    # Save level-wise statistics
    stats_output = output_file.replace('.csv', '_level_stats.csv')
    level_stats.to_csv(stats_output)
    print(f"Level-wise statistics saved to: {stats_output}")

if __name__ == '__main__':
    main()
