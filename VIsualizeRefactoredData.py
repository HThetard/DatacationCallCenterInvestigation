"""
Visualize Refactored Call Center Data
======================================
This script creates comprehensive visualizations comparing original and
refactored call center data, focusing on:
- IncomingCalls over time
- ActualWorkloadMinutes over time
- Comparison of outlier handling methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


class DataVisualizer:
    """Visualize original and refactored call center data."""
    
    def __init__(self):
        """Load all datasets."""
        print("Loading datasets...")
        
        # Load original data
        self.df_original = pd.read_excel('CallCenterData.xlsx')
        
        # Load refactored datasets
        self.datasets = {
            'Original': self.df_original
        }
        
        # Find all refactored files
        refactored_files = glob.glob('CallCenterData_*_*.xlsx')
        for file in refactored_files:
            # Extract method name from filename
            method_name = file.replace('CallCenterData_', '').split('_2026')[0]
            method_display = method_name.replace('_', ' ').title()
            self.datasets[method_display] = pd.read_excel(file)
            print(f"  Loaded: {method_display}")
        
        # Ensure Date column is datetime
        for name, df in self.datasets.items():
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            self.datasets[name] = df
        
        print(f"\nTotal datasets loaded: {len(self.datasets)}\n")
    
    def plot_time_series_comparison(self):
        """Create time series plots with Date on X-axis, IncomingCalls and ActualWorkloadMinutes."""
        print("Creating time series comparison plots...")
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        
        # Plot 1: IncomingCalls over time
        ax1 = axes[0]
        for name, df in self.datasets.items():
            if name == 'Original':
                ax1.plot(df['Date'], df['IncomingCalls'], 
                        label=name, linewidth=2.5, alpha=0.7, color='black', linestyle='--')
            else:
                ax1.plot(df['Date'], df['IncomingCalls'], 
                        label=name, linewidth=1.5, alpha=0.8)
        
        ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Incoming Calls', fontsize=12, fontweight='bold')
        ax1.set_title('Incoming Calls Over Time - Original vs Refactored Data', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: ActualWorkloadMinutes over time
        ax2 = axes[1]
        for name, df in self.datasets.items():
            if name == 'Original':
                ax2.plot(df['Date'], df['ActualWorkloadMinutes'], 
                        label=name, linewidth=2.5, alpha=0.7, color='black', linestyle='--')
            else:
                ax2.plot(df['Date'], df['ActualWorkloadMinutes'], 
                        label=name, linewidth=1.5, alpha=0.8)
        
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Actual Workload (Minutes)', fontsize=12, fontweight='bold')
        ax2.set_title('Actual Workload Minutes Over Time - Original vs Refactored Data', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('TimeSeriesComparison.png', dpi=300, bbox_inches='tight')
        print("  Saved: TimeSeriesComparison.png")
        plt.show()
    
    def plot_dual_axis(self):
        """Create dual-axis plot with IncomingCalls and ActualWorkloadMinutes."""
        print("\nCreating dual-axis plots for each method...")
        
        num_datasets = len(self.datasets)
        fig, axes = plt.subplots(num_datasets, 1, figsize=(16, 6 * num_datasets))
        
        if num_datasets == 1:
            axes = [axes]
        
        for idx, (name, df) in enumerate(self.datasets.items()):
            ax1 = axes[idx]
            
            # Plot IncomingCalls on left y-axis
            color1 = 'tab:blue'
            ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Incoming Calls', color=color1, fontsize=12, fontweight='bold')
            line1 = ax1.plot(df['Date'], df['IncomingCalls'], 
                           color=color1, linewidth=2, label='Incoming Calls', marker='o', markersize=3)
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True, alpha=0.3)
            
            # Create second y-axis for ActualWorkloadMinutes
            ax2 = ax1.twinx()
            color2 = 'tab:orange'
            ax2.set_ylabel('Actual Workload (Minutes)', color=color2, fontsize=12, fontweight='bold')
            line2 = ax2.plot(df['Date'], df['ActualWorkloadMinutes'], 
                           color=color2, linewidth=2, label='Actual Workload Minutes', marker='s', markersize=3)
            ax2.tick_params(axis='y', labelcolor=color2)
            
            # Title
            ax1.set_title(f'{name} Dataset: Incoming Calls vs Actual Workload Minutes Over Time', 
                         fontsize=14, fontweight='bold', pad=20)
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left', framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig('DualAxisComparison.png', dpi=300, bbox_inches='tight')
        print("  Saved: DualAxisComparison.png")
        plt.show()
    
    def plot_scatter_comparison(self):
        """Create scatter plots showing relationship between IncomingCalls and ActualWorkloadMinutes."""
        print("\nCreating scatter plot comparison...")
        
        num_datasets = len(self.datasets)
        cols = 3
        rows = (num_datasets + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
        axes = axes.flatten() if num_datasets > 1 else [axes]
        
        for idx, (name, df) in enumerate(self.datasets.items()):
            ax = axes[idx]
            
            # Create scatter plot
            scatter = ax.scatter(df['IncomingCalls'], df['ActualWorkloadMinutes'], 
                               alpha=0.6, s=50, c=df.index, cmap='viridis')
            
            # Add trend line
            z = np.polyfit(df['IncomingCalls'], df['ActualWorkloadMinutes'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df['IncomingCalls'].min(), df['IncomingCalls'].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
            
            # Calculate correlation
            corr = df['IncomingCalls'].corr(df['ActualWorkloadMinutes'])
            
            ax.set_xlabel('Incoming Calls', fontsize=11, fontweight='bold')
            ax.set_ylabel('Actual Workload (Minutes)', fontsize=11, fontweight='bold')
            ax.set_title(f'{name}\nCorrelation: {corr:.3f}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Time Sequence', rotation=270, labelpad=15)
        
        # Hide extra subplots
        for idx in range(len(self.datasets), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('ScatterComparison.png', dpi=300, bbox_inches='tight')
        print("  Saved: ScatterComparison.png")
        plt.show()
    
    def plot_distribution_comparison(self):
        """Create distribution plots for key metrics."""
        print("\nCreating distribution comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = ['IncomingCalls', 'ActualWorkloadMinutes']
        
        for col_idx, metric in enumerate(metrics):
            # Histogram
            ax_hist = axes[0, col_idx]
            for name, df in self.datasets.items():
                ax_hist.hist(df[metric], bins=30, alpha=0.5, label=name, edgecolor='black')
            ax_hist.set_xlabel(metric, fontsize=11, fontweight='bold')
            ax_hist.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax_hist.set_title(f'Distribution of {metric}', fontsize=12, fontweight='bold')
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.3)
            
            # Box plot
            ax_box = axes[1, col_idx]
            data_to_plot = [df[metric] for df in self.datasets.values()]
            box = ax_box.boxplot(data_to_plot, labels=list(self.datasets.keys()), patch_artist=True)
            
            # Color boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.datasets)))
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
            
            ax_box.set_ylabel(metric, fontsize=11, fontweight='bold')
            ax_box.set_title(f'Box Plot of {metric}', fontsize=12, fontweight='bold')
            ax_box.tick_params(axis='x', rotation=45)
            ax_box.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('DistributionComparison.png', dpi=300, bbox_inches='tight')
        print("  Saved: DistributionComparison.png")
        plt.show()
    
    def plot_summary_statistics(self):
        """Create a summary statistics comparison table visualization."""
        print("\nCreating summary statistics comparison...")
        
        metrics = ['IncomingCalls', 'ActualWorkloadMinutes']
        
        for metric in metrics:
            summary_data = []
            
            for name, df in self.datasets.items():
                summary_data.append({
                    'Method': name,
                    'Mean': df[metric].mean(),
                    'Median': df[metric].median(),
                    'Std Dev': df[metric].std(),
                    'Min': df[metric].min(),
                    'Max': df[metric].max(),
                    'Q1': df[metric].quantile(0.25),
                    'Q3': df[metric].quantile(0.75)
                })
            
            summary_df = pd.DataFrame(summary_data)
            
            # Create table visualization
            fig, ax = plt.subplots(figsize=(14, len(self.datasets) * 0.5 + 1))
            ax.axis('tight')
            ax.axis('off')
            
            table_data = []
            for _, row in summary_df.iterrows():
                table_data.append([
                    row['Method'],
                    f"{row['Mean']:.2f}",
                    f"{row['Median']:.2f}",
                    f"{row['Std Dev']:.2f}",
                    f"{row['Min']:.2f}",
                    f"{row['Max']:.2f}",
                    f"{row['Q1']:.2f}",
                    f"{row['Q3']:.2f}"
                ])
            
            table = ax.table(cellText=table_data,
                           colLabels=['Method', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3'],
                           cellLoc='center',
                           loc='center',
                           bbox=[0, 0, 1, 1])
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Color header row
            for i in range(8):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color alternating rows
            for i in range(1, len(table_data) + 1):
                for j in range(8):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f0f0f0')
            
            plt.title(f'Summary Statistics Comparison - {metric}', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.savefig(f'SummaryStats_{metric}.png', dpi=300, bbox_inches='tight')
            print(f"  Saved: SummaryStats_{metric}.png")
            plt.show()
    
    def create_all_visualizations(self):
        """Generate all visualization types."""
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        self.plot_time_series_comparison()
        self.plot_dual_axis()
        self.plot_scatter_comparison()
        self.plot_distribution_comparison()
        self.plot_summary_statistics()
        
        print("\n" + "=" * 70)
        print("All visualizations completed successfully!")
        print("=" * 70)


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("CALL CENTER DATA VISUALIZATION")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    visualizer = DataVisualizer()
    visualizer.create_all_visualizations()
    
    print("\n📊 Visualization Summary:")
    print("  1. TimeSeriesComparison.png - IncomingCalls & ActualWorkloadMinutes over time")
    print("  2. DualAxisComparison.png - Dual-axis plots for each method")
    print("  3. ScatterComparison.png - Relationship between metrics")
    print("  4. DistributionComparison.png - Distribution histograms and box plots")
    print("  5. SummaryStats_*.png - Statistical comparison tables")
    print("\nAll visualizations saved to current directory!")


if __name__ == "__main__":
    main()
