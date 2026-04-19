#!/usr/bin/env python3
"""
Simple visualization tools for experiments
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def create_summary_plots(all_results, output_dir):
    """
    Create summary plots from experiment results
    """
    if not all_results:
        print("Warning: No results to visualize")
        return
    
    try:
        # Prepare data
        data = []
        for result in all_results:
            data.append({
                'Algorithm': result['algorithm'],
                'Environment': result['env'],
                'Seed': str(result['seed']),
                'Avg Reward': result['evaluation']['avg_reward'],
                'Success Rate': result['evaluation']['success_rate'],
                'Comm Cost': result['evaluation']['comm_cost']
            })
        
        df = pd.DataFrame(data)
        
        # Create output directory
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Average reward comparison
        plt.figure(figsize=(10, 6))
        algorithms = df['Algorithm'].unique()
        avg_rewards = [df[df['Algorithm'] == algo]['Avg Reward'].mean() for algo in algorithms]
        std_rewards = [df[df['Algorithm'] == algo]['Avg Reward'].std() for algo in algorithms]
        
        x_pos = np.arange(len(algorithms))
        plt.bar(x_pos, avg_rewards, yerr=std_rewards, capsize=10, alpha=0.7)
        plt.xticks(x_pos, algorithms, rotation=45)
        plt.title('Average Reward Comparison')
        plt.xlabel('Algorithm')
        plt.ylabel('Average Reward')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'avg_reward_comparison.png'), dpi=150)
        plt.close()
        
        # 2. Success rate comparison
        plt.figure(figsize=(10, 6))
        success_rates = [df[df['Algorithm'] == algo]['Success Rate'].mean() for algo in algorithms]
        
        plt.bar(x_pos, success_rates, alpha=0.7)
        plt.xticks(x_pos, algorithms, rotation=45)
        plt.title('Success Rate Comparison')
        plt.xlabel('Algorithm')
        plt.ylabel('Success Rate')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'success_rate_comparison.png'), dpi=150)
        plt.close()
        
        # 3. Communication cost comparison
        plt.figure(figsize=(10, 6))
        comm_costs = [df[df['Algorithm'] == algo]['Comm Cost'].mean() for algo in algorithms]
        
        plt.bar(x_pos, comm_costs, alpha=0.7)
        plt.xticks(x_pos, algorithms, rotation=45)
        plt.title('Communication Cost Comparison')
        plt.xlabel('Algorithm')
        plt.ylabel('Comm Cost (KB/step)')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'comm_cost_comparison.png'), dpi=150)
        plt.close()
        
        print(f"Simple plots saved to: {plots_dir}")
        
    except Exception as e:
        print(f"Warning: Could not create visualizations: {e}")

def create_comprehensive_report(all_results, output_dir):
    """
    Create comprehensive experiment report
    """
    if not all_results:
        return
    
    try:
        # Create report directory
        report_dir = os.path.join(output_dir, 'report')
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate summary table
        summary_data = []
        for result in all_results:
            summary_data.append({
                'Algorithm': result['algorithm'],
                'Environment': result['env'],
                'Seed': str(result['seed']),
                'Avg Reward': f"{result['evaluation']['avg_reward']:.2f}",
                'Success Rate': f"{result['evaluation']['success_rate']:.2%}",
                'Comm Cost': f"{result['evaluation']['comm_cost']:.2f} KB/step"
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Save as CSV
        csv_path = os.path.join(report_dir, 'experiment_summary.csv')
        df_summary.to_csv(csv_path, index=False)
        
        # Save as text file
        txt_path = os.path.join(report_dir, 'experiment_summary.txt')
        with open(txt_path, 'w') as f:
            f.write('Experiment Summary Report\n')
            f.write('=' * 50 + '\n\n')
            
            for algo in df_summary['Algorithm'].unique():
                algo_data = df_summary[df_summary['Algorithm'] == algo]
                f.write(f'Algorithm: {algo}\n')
                f.write('-' * 30 + '\n')
                
                for _, row in algo_data.iterrows():
                    f.write(f"  Seed {row['Seed']}: ")
                    f.write(f"Reward={row['Avg Reward']}, ")
                    f.write(f"Success={row['Success Rate']}, ")
                    f.write(f"Comm Cost={row['Comm Cost']}\n")
                f.write('\n')
        
        print(f"Report saved to: {report_dir}")
        
    except Exception as e:
        print(f"Warning: Could not create report: {e}")

if __name__ == "__main__":
    # Test the visualization tool
    test_results = [
        {
            'algorithm': 'MAPPO',
            'env': 'MPE_Navigation',
            'seed': 42,
            'evaluation': {
                'avg_reward': 85.3,
                'success_rate': 0.88,
                'comm_cost': 4.5
            }
        },
        {
            'algorithm': 'IACN',
            'env': 'MPE_Navigation',
            'seed': 42,
            'evaluation': {
                'avg_reward': 75.1,
                'success_rate': 0.75,
                'comm_cost': 8.8
            }
        }
    ]
    
    create_summary_plots(test_results, 'test_output')
    create_comprehensive_report(test_results, 'test_output')
    print("Test completed!")