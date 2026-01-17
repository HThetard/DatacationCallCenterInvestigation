"""
Call Center Forecasting with Prophet
=====================================
This script uses Facebook Prophet to forecast IncomingCalls for the call center.
- Training period: 2020-2022
- Forecast horizon: 2 years into the future
- Comparison: Training actuals vs forecasted values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


class CallCenterForecaster:
    """Forecast call center incoming calls using Prophet."""
    
    def __init__(self, data_path):
        """Load and prepare the z-score filtered dataset."""
        print("=" * 70)
        print("CALL CENTER FORECASTING WITH PROPHET")
        print("=" * 70)
        print(f"Loading data from: {data_path}\n")
        
        # Load data
        self.df = pd.read_excel(data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        print(f"Data loaded successfully!")
        print(f"  Total records: {len(self.df)}")
        print(f"  Date range: {self.df['Date'].min().date()} to {self.df['Date'].max().date()}")
        print(f"  Years covered: {self.df['Date'].dt.year.unique()}")
        print()
    
    def prepare_training_data(self, start_year=2020, end_year=2022):
        """
        Prepare training dataset for Prophet.
        Prophet requires columns named 'ds' (date) and 'y' (value).
        """
        print("=" * 70)
        print("PREPARING TRAINING DATA")
        print("=" * 70)
        print(f"Training period: {start_year} - {end_year}")
        
        # Filter data for training period
        train_mask = (self.df['Date'].dt.year >= start_year) & (self.df['Date'].dt.year <= end_year)
        self.train_df = self.df[train_mask].copy()
        
        # Prepare for Prophet (rename columns)
        self.prophet_train = pd.DataFrame({
            'ds': self.train_df['Date'],
            'y': self.train_df['IncomingCalls']
        })
        
        print(f"\nTraining data prepared:")
        print(f"  Records: {len(self.prophet_train)}")
        print(f"  Date range: {self.prophet_train['ds'].min().date()} to {self.prophet_train['ds'].max().date()}")
        print(f"  Mean IncomingCalls: {self.prophet_train['y'].mean():.2f}")
        print(f"  Std Dev: {self.prophet_train['y'].std():.2f}")
        print()
        
        # Store test data (if exists beyond training period)
        test_mask = self.df['Date'].dt.year > end_year
        self.test_df = self.df[test_mask].copy() if test_mask.sum() > 0 else None
        
        if self.test_df is not None and len(self.test_df) > 0:
            print(f"Test data available:")
            print(f"  Records: {len(self.test_df)}")
            print(f"  Date range: {self.test_df['Date'].min().date()} to {self.test_df['Date'].max().date()}")
            print()
    
    def train_prophet_model(self):
        """Train the Prophet model."""
        print("=" * 70)
        print("TRAINING PROPHET MODEL")
        print("=" * 70)
        
        # Initialize Prophet with optimized parameters for call center data
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='additive',
            changepoint_prior_scale=0.05,  # Controls trend flexibility
            seasonality_prior_scale=10.0    # Controls seasonality flexibility
        )
        
        print("Model configuration:")
        print("  Yearly seasonality: Enabled")
        print("  Weekly seasonality: Enabled")
        print("  Seasonality mode: Additive")
        print("\nTraining model...")
        
        # Fit the model
        self.model.fit(self.prophet_train)
        
        print("✓ Model trained successfully!")
        print()
    
    def generate_forecast(self, periods=730):
        """
        Generate forecast for specified number of days.
        Default: 730 days = 2 years
        """
        print("=" * 70)
        print("GENERATING FORECAST")
        print("=" * 70)
        print(f"Forecast horizon: {periods} days ({periods/365:.1f} years)")
        
        # Create future dataframe
        self.future = self.model.make_future_dataframe(periods=periods, freq='D')
        
        print(f"Forecasting from {self.future['ds'].min().date()} to {self.future['ds'].max().date()}")
        print("Generating predictions...")
        
        # Generate predictions
        self.forecast = self.model.predict(self.future)
        
        print("✓ Forecast completed!")
        print(f"  Total predictions: {len(self.forecast)}")
        print(f"  Training period predictions: {len(self.prophet_train)}")
        print(f"  Future predictions: {periods}")
        print()
    
    def plot_forecast_comparison(self):
        """Create comprehensive visualization comparing actuals vs forecast."""
        print("Creating forecast visualization...")
        
        fig, axes = plt.subplots(3, 1, figsize=(18, 14))
        
        # Plot 1: Full overview - Training data and forecast
        ax1 = axes[0]
        
        # Plot actual training data
        ax1.plot(self.prophet_train['ds'], self.prophet_train['y'], 
                'o-', color='black', linewidth=2, markersize=3, 
                label='Actual (Training 2020-2022)', zorder=5)
        
        # Plot forecast
        ax1.plot(self.forecast['ds'], self.forecast['yhat'], 
                '-', color='blue', linewidth=2, 
                label='Forecast', alpha=0.8)
        
        # Plot confidence intervals
        ax1.fill_between(self.forecast['ds'], 
                         self.forecast['yhat_lower'], 
                         self.forecast['yhat_upper'],
                         color='blue', alpha=0.2, 
                         label='95% Confidence Interval')
        
        # Mark the forecast start
        forecast_start = self.prophet_train['ds'].max()
        ax1.axvline(x=forecast_start, color='red', linestyle='--', 
                   linewidth=2, label='Forecast Start')
        
        ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Incoming Calls', fontsize=12, fontweight='bold')
        ax1.set_title('Prophet Forecast: Incoming Calls (Full View)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Zoomed on training period
        ax2 = axes[1]
        
        # Filter forecast for training period
        train_forecast = self.forecast[self.forecast['ds'] <= forecast_start]
        
        ax2.plot(self.prophet_train['ds'], self.prophet_train['y'], 
                'o', color='black', markersize=4, 
                label='Actual Values', zorder=5)
        ax2.plot(train_forecast['ds'], train_forecast['yhat'], 
                '-', color='green', linewidth=2, 
                label='Fitted Values', alpha=0.8)
        ax2.fill_between(train_forecast['ds'], 
                         train_forecast['yhat_lower'], 
                         train_forecast['yhat_upper'],
                         color='green', alpha=0.2)
        
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Incoming Calls', fontsize=12, fontweight='bold')
        ax2.set_title('Model Fit on Training Data (2020-2022)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Zoomed on forecast period only
        ax3 = axes[2]
        
        # Filter for forecast period only
        future_forecast = self.forecast[self.forecast['ds'] > forecast_start]
        
        ax3.plot(future_forecast['ds'], future_forecast['yhat'], 
                '-', color='blue', linewidth=2.5, 
                label='Forecast (2 years)', marker='o', markersize=2)
        ax3.fill_between(future_forecast['ds'], 
                         future_forecast['yhat_lower'], 
                         future_forecast['yhat_upper'],
                         color='blue', alpha=0.3, 
                         label='95% Confidence Interval')
        
        ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Incoming Calls', fontsize=12, fontweight='bold')
        ax3.set_title('Future Forecast (2 Years Ahead)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Prophet_Forecast_Comparison.png', dpi=300, bbox_inches='tight')
        print("  Saved: Prophet_Forecast_Comparison.png")
        plt.show()
    
    def plot_components(self):
        """Plot Prophet's forecast components (trend, seasonality)."""
        print("\nCreating component plots...")
        
        fig = self.model.plot_components(self.forecast, figsize=(16, 10))
        plt.savefig('Prophet_Components.png', dpi=300, bbox_inches='tight')
        print("  Saved: Prophet_Components.png")
        plt.show()
    
    def calculate_metrics(self):
        """Calculate forecast accuracy metrics on training period."""
        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE METRICS (Training Period)")
        print("=" * 70)
        
        # Get predictions for training period
        train_forecast = self.forecast[self.forecast['ds'].isin(self.prophet_train['ds'])]
        
        # Merge with actuals
        comparison = self.prophet_train.merge(
            train_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
            on='ds'
        )
        
        # Calculate metrics
        comparison['error'] = comparison['y'] - comparison['yhat']
        comparison['abs_error'] = np.abs(comparison['error'])
        comparison['pct_error'] = (comparison['error'] / comparison['y']) * 100
        comparison['abs_pct_error'] = np.abs(comparison['pct_error'])
        
        mae = comparison['abs_error'].mean()
        rmse = np.sqrt((comparison['error'] ** 2).mean())
        mape = comparison['abs_pct_error'].mean()
        
        print(f"\nAccuracy Metrics:")
        print(f"  Mean Absolute Error (MAE): {mae:.2f} calls")
        print(f"  Root Mean Square Error (RMSE): {rmse:.2f} calls")
        print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"\nError Statistics:")
        print(f"  Mean Error: {comparison['error'].mean():.2f}")
        print(f"  Std Dev of Errors: {comparison['error'].std():.2f}")
        print(f"  Min Error: {comparison['error'].min():.2f}")
        print(f"  Max Error: {comparison['error'].max():.2f}")
        
        # Calculate coverage (% of actuals within confidence interval)
        within_interval = (
            (comparison['y'] >= comparison['yhat_lower']) & 
            (comparison['y'] <= comparison['yhat_upper'])
        ).sum()
        coverage = (within_interval / len(comparison)) * 100
        print(f"\nConfidence Interval Coverage: {coverage:.1f}%")
        print()
        
        return comparison
    
    def export_results(self, comparison):
        """Export forecast results to Excel."""
        print("Exporting forecast results...")
        
        # Prepare full forecast export
        forecast_export = self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_export.columns = ['Date', 'Forecast', 'Lower_Bound', 'Upper_Bound']
        
        # Add actual values where available
        forecast_export = forecast_export.merge(
            self.df[['Date', 'IncomingCalls']], 
            on='Date', 
            how='left'
        )
        forecast_export.rename(columns={'IncomingCalls': 'Actual'}, inplace=True)
        
        # Add flag for training vs forecast period
        forecast_start = self.prophet_train['ds'].max()
        forecast_export['Period'] = forecast_export['Date'].apply(
            lambda x: 'Training' if x <= forecast_start else 'Forecast'
        )
        
        # Reorder columns
        forecast_export = forecast_export[['Date', 'Actual', 'Forecast', 
                                          'Lower_Bound', 'Upper_Bound', 'Period']]
        
        # Export to Excel
        filename = 'Prophet_Forecast_Results.xlsx'
        
        with pd.ExcelWriter(filename) as writer:
            forecast_export.to_excel(writer, sheet_name='Full_Forecast', index=False)
            comparison.to_excel(writer, sheet_name='Training_Comparison', index=False)
        
        print(f"  Saved: {filename}")
        print(f"    - Full_Forecast sheet: {len(forecast_export)} rows")
        print(f"    - Training_Comparison sheet: {len(comparison)} rows")
        print()
    
    def run_full_forecast(self, train_start=2020, train_end=2022, forecast_years=2):
        """
        Run complete forecasting workflow.
        
        Parameters:
        - train_start: Starting year for training
        - train_end: Ending year for training
        - forecast_years: Number of years to forecast into the future
        """
        self.prepare_training_data(train_start, train_end)
        self.train_prophet_model()
        self.generate_forecast(periods=forecast_years * 365)
        self.plot_forecast_comparison()
        self.plot_components()
        comparison = self.calculate_metrics()
        self.export_results(comparison)
        
        print("=" * 70)
        print("FORECASTING COMPLETE!")
        print("=" * 70)
        print("\n📊 Generated Files:")
        print("  1. Prophet_Forecast_Comparison.png - Training vs Forecast comparison")
        print("  2. Prophet_Components.png - Trend and seasonality components")
        print("  3. Prophet_Forecast_Results.xlsx - Full forecast data")
        print("\n✓ Analysis completed successfully!")


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("CALL CENTER FORECASTING WITH PROPHET")
    print("=" * 70)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    
    # Initialize forecaster with z-score filtered data
    forecaster = CallCenterForecaster('CallCenterData_zscore_filtered_20260117_231525.xlsx')
    
    # Run full forecast: Train on 2020-2022, forecast 2 years ahead
    forecaster.run_full_forecast(train_start=2020, train_end=2022, forecast_years=2)


if __name__ == "__main__":
    main()
