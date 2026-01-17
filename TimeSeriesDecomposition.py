"""
Time Series Decomposition for Call Center Data
===============================================
This script performs time series decomposition on the IncomingCalls variable
from the z-score filtered dataset to identify:
- Trend component
- Seasonal component  
- Residual component
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


class TimeSeriesDecomposer:
    """Decompose time series data for call center incoming calls."""
    
    def __init__(self, data_path):
        """Load the z-score filtered dataset."""
        print("=" * 70)
        print("TIME SERIES DECOMPOSITION - INCOMING CALLS")
        print("=" * 70)
        print(f"Loading data from: {data_path}\n")
        
        self.df = pd.read_excel(data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Set Date as index for time series analysis
        self.ts = self.df.set_index('Date')['IncomingCalls']
        
        print(f"Data loaded successfully!")
        print(f"  Date range: {self.ts.index.min().date()} to {self.ts.index.max().date()}")
        print(f"  Total observations: {len(self.ts)}")
        print(f"  Frequency: {self.ts.index.inferred_freq or 'Daily'}")
        print()
    
    def check_stationarity(self):
        """Perform Augmented Dickey-Fuller test for stationarity."""
        print("=" * 70)
        print("STATIONARITY TEST (Augmented Dickey-Fuller)")
        print("=" * 70)
        
        result = adfuller(self.ts.dropna())
        
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"P-value: {result[1]:.4f}")
        print(f"Critical Values:")
        for key, value in result[4].items():
            print(f"  {key}: {value:.4f}")
        
        if result[1] <= 0.05:
            print("\n✓ Series is STATIONARY (p-value <= 0.05)")
        else:
            print("\n✗ Series is NON-STATIONARY (p-value > 0.05)")
            print("  Decomposition will help identify trend and seasonal patterns.")
        print()
    
    def perform_decomposition(self, model='additive', period=7):
        """
        Perform seasonal decomposition.
        
        Parameters:
        - model: 'additive' or 'multiplicative'
        - period: seasonal period (7 for weekly, 30 for monthly)
        """
        print("=" * 70)
        print(f"SEASONAL DECOMPOSITION ({model.upper()} MODEL)")
        print("=" * 70)
        print(f"Seasonal period: {period} days")
        print()
        
        # Perform decomposition
        self.decomposition = seasonal_decompose(
            self.ts, 
            model=model, 
            period=period,
            extrapolate_trend='freq'
        )
        
        # Extract components
        self.trend = self.decomposition.trend
        self.seasonal = self.decomposition.seasonal
        self.residual = self.decomposition.resid
        
        # Calculate statistics for each component
        print("Component Statistics:")
        print("-" * 70)
        
        components = {
            'Original': self.ts,
            'Trend': self.trend,
            'Seasonal': self.seasonal,
            'Residual': self.residual
        }
        
        for name, component in components.items():
            if component is not None:
                print(f"\n{name}:")
                print(f"  Mean: {component.mean():.2f}")
                print(f"  Std Dev: {component.std():.2f}")
                print(f"  Min: {component.min():.2f}")
                print(f"  Max: {component.max():.2f}")
        
        print()
    
    def plot_decomposition(self):
        """Create comprehensive decomposition visualizations."""
        print("Creating decomposition plots...")
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        
        # Original series
        axes[0].plot(self.ts.index, self.ts.values, color='steelblue', linewidth=1.5)
        axes[0].set_ylabel('Incoming Calls', fontsize=11, fontweight='bold')
        axes[0].set_title('Original Time Series - Incoming Calls', 
                         fontsize=13, fontweight='bold', pad=10)
        axes[0].grid(True, alpha=0.3)
        
        # Trend component
        axes[1].plot(self.trend.index, self.trend.values, color='red', linewidth=2)
        axes[1].set_ylabel('Trend', fontsize=11, fontweight='bold')
        axes[1].set_title('Trend Component', fontsize=13, fontweight='bold', pad=10)
        axes[1].grid(True, alpha=0.3)
        
        # Seasonal component
        axes[2].plot(self.seasonal.index, self.seasonal.values, color='green', linewidth=1.5)
        axes[2].set_ylabel('Seasonal', fontsize=11, fontweight='bold')
        axes[2].set_title('Seasonal Component', fontsize=13, fontweight='bold', pad=10)
        axes[2].grid(True, alpha=0.3)
        
        # Residual component
        axes[3].plot(self.residual.index, self.residual.values, color='purple', linewidth=1, alpha=0.7)
        axes[3].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[3].set_ylabel('Residual', fontsize=11, fontweight='bold')
        axes[3].set_xlabel('Date', fontsize=11, fontweight='bold')
        axes[3].set_title('Residual Component', fontsize=13, fontweight='bold', pad=10)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('TimeSeriesDecomposition.png', dpi=300, bbox_inches='tight')
        print("  Saved: TimeSeriesDecomposition.png")
        plt.show()
    
    def plot_seasonal_pattern(self):
        """Visualize the seasonal pattern more clearly."""
        print("\nCreating seasonal pattern analysis...")
        
        # Extract day of week from seasonal component
        seasonal_df = pd.DataFrame({
            'Date': self.seasonal.index,
            'Seasonal': self.seasonal.values
        })
        seasonal_df['DayOfWeek'] = seasonal_df['Date'].dt.day_name()
        seasonal_df['DayNum'] = seasonal_df['Date'].dt.dayofweek
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Seasonal pattern over time with day markers
        ax1 = axes[0]
        ax1.plot(seasonal_df['Date'], seasonal_df['Seasonal'], 
                linewidth=1.5, color='green', alpha=0.7)
        ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Seasonal Component', fontsize=11, fontweight='bold')
        ax1.set_title('Seasonal Pattern Over Time', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
        
        # Plot 2: Average seasonal effect by day of week
        ax2 = axes[1]
        day_avg = seasonal_df.groupby('DayOfWeek')['Seasonal'].mean()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_avg = day_avg.reindex(day_order)
        
        colors = ['green' if x > 0 else 'red' for x in day_avg.values]
        bars = ax2.bar(range(len(day_avg)), day_avg.values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(len(day_avg)))
        ax2.set_xticklabels(day_avg.index, rotation=45, ha='right')
        ax2.set_ylabel('Average Seasonal Effect', fontsize=11, fontweight='bold')
        ax2.set_title('Average Seasonal Effect by Day of Week', fontsize=13, fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, day_avg.values)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('SeasonalPattern.png', dpi=300, bbox_inches='tight')
        print("  Saved: SeasonalPattern.png")
        plt.show()
    
    def plot_acf_pacf(self):
        """Plot autocorrelation and partial autocorrelation functions."""
        print("\nCreating ACF and PACF plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # ACF for original series
        acf_vals = acf(self.ts.dropna(), nlags=40)
        axes[0, 0].stem(range(len(acf_vals)), acf_vals, basefmt=' ')
        axes[0, 0].axhline(y=0, color='black', linewidth=0.8)
        axes[0, 0].axhline(y=1.96/np.sqrt(len(self.ts)), color='red', linestyle='--', linewidth=1)
        axes[0, 0].axhline(y=-1.96/np.sqrt(len(self.ts)), color='red', linestyle='--', linewidth=1)
        axes[0, 0].set_xlabel('Lag', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('ACF', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Autocorrelation Function - Original Series', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # PACF for original series
        pacf_vals = pacf(self.ts.dropna(), nlags=40)
        axes[0, 1].stem(range(len(pacf_vals)), pacf_vals, basefmt=' ')
        axes[0, 1].axhline(y=0, color='black', linewidth=0.8)
        axes[0, 1].axhline(y=1.96/np.sqrt(len(self.ts)), color='red', linestyle='--', linewidth=1)
        axes[0, 1].axhline(y=-1.96/np.sqrt(len(self.ts)), color='red', linestyle='--', linewidth=1)
        axes[0, 1].set_xlabel('Lag', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('PACF', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Partial Autocorrelation Function - Original Series', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # ACF for residuals
        residual_clean = self.residual.dropna()
        acf_resid = acf(residual_clean, nlags=40)
        axes[1, 0].stem(range(len(acf_resid)), acf_resid, basefmt=' ')
        axes[1, 0].axhline(y=0, color='black', linewidth=0.8)
        axes[1, 0].axhline(y=1.96/np.sqrt(len(residual_clean)), color='red', linestyle='--', linewidth=1)
        axes[1, 0].axhline(y=-1.96/np.sqrt(len(residual_clean)), color='red', linestyle='--', linewidth=1)
        axes[1, 0].set_xlabel('Lag', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('ACF', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Autocorrelation Function - Residuals', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residual histogram
        axes[1, 1].hist(residual_clean, bins=30, edgecolor='black', alpha=0.7, color='purple')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Residual Value', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('ACF_PACF_Analysis.png', dpi=300, bbox_inches='tight')
        print("  Saved: ACF_PACF_Analysis.png")
        plt.show()
    
    def analyze_residuals(self):
        """Analyze residual component for randomness."""
        print("\n" + "=" * 70)
        print("RESIDUAL ANALYSIS")
        print("=" * 70)
        
        residual_clean = self.residual.dropna()
        
        # Test for normality (Shapiro-Wilk would be too slow for large datasets)
        print(f"Residual Statistics:")
        print(f"  Mean: {residual_clean.mean():.4f} (should be ~0)")
        print(f"  Std Dev: {residual_clean.std():.4f}")
        print(f"  Skewness: {residual_clean.skew():.4f} (should be ~0)")
        print(f"  Kurtosis: {residual_clean.kurtosis():.4f} (should be ~0)")
        
        # Check for autocorrelation in residuals
        result = adfuller(residual_clean)
        print(f"\nADF test on residuals:")
        print(f"  P-value: {result[1]:.4f}")
        
        if result[1] <= 0.05:
            print("  ✓ Residuals appear to be stationary (white noise)")
        else:
            print("  ✗ Residuals may have remaining structure")
        
        print()
    
    def export_components(self):
        """Export decomposed components to Excel."""
        print("Exporting decomposed components...")
        
        export_df = pd.DataFrame({
            'Date': self.ts.index,
            'Original': self.ts.values,
            'Trend': self.trend.values,
            'Seasonal': self.seasonal.values,
            'Residual': self.residual.values
        })
        
        filename = 'IncomingCalls_Decomposed_Components.xlsx'
        export_df.to_excel(filename, index=False)
        print(f"  Saved: {filename}")
        print()
    
    def run_full_analysis(self, period=7, model='additive'):
        """
        Run complete decomposition analysis.
        
        Parameters:
        - period: seasonal period (7 for weekly, 30 for monthly)
        - model: 'additive' or 'multiplicative'
        """
        self.check_stationarity()
        self.perform_decomposition(model=model, period=period)
        self.plot_decomposition()
        self.plot_seasonal_pattern()
        self.plot_acf_pacf()
        self.analyze_residuals()
        self.export_components()
        
        print("=" * 70)
        print("DECOMPOSITION COMPLETE!")
        print("=" * 70)
        print("\n📊 Generated Files:")
        print("  1. TimeSeriesDecomposition.png - Full decomposition plot")
        print("  2. SeasonalPattern.png - Seasonal pattern analysis")
        print("  3. ACF_PACF_Analysis.png - Autocorrelation analysis")
        print("  4. IncomingCalls_Decomposed_Components.xlsx - Component data")
        print("\n✓ Analysis completed successfully!")


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("CALL CENTER TIME SERIES DECOMPOSITION")
    print("=" * 70)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    
    # Initialize decomposer with z-score filtered data
    decomposer = TimeSeriesDecomposer('CallCenterData_zscore_filtered_20260117_231525.xlsx')
    
    # Run full analysis with weekly seasonality (period=7)
    decomposer.run_full_analysis(period=7, model='additive')


if __name__ == "__main__":
    main()
