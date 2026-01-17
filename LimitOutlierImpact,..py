"""
Call Center Data Outlier Impact Reduction
==========================================
This script analyzes call center data and applies various techniques to reduce
the impact of outliers on statistical analysis and modeling.

Techniques used:
1. Winsorization (capping outliers at percentiles)
2. IQR-based capping
3. Log transformation for skewed distributions
4. Z-score based filtering
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class OutlierHandler:
    """
    A comprehensive class for detecting and handling outliers in call center data.
    """
    
    def __init__(self, data_path):
        """Initialize with path to Excel file."""
        self.df = pd.read_excel(data_path)
        self.df_original = self.df.copy()
        self.numerical_cols = [
            'IncomingCalls', 'AnsweredCalls', 'AnswerRate', 'AbandonedCalls',
            'ServiceLevel(20Seconds)', 'ActualWorkloadhours', 'ActualWorkloadMinutes',
            'RequiredWorkloadHours', 'RequiredWorkloadMinutes'
        ]
        
    def analyze_outliers(self):
        """Detect outliers using IQR method."""
        print("=" * 70)
        print("OUTLIER DETECTION REPORT")
        print("=" * 70)
        
        outlier_summary = []
        
        for col in self.numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_count = outliers_mask.sum()
            outlier_pct = (outlier_count / len(self.df)) * 100
            
            outlier_summary.append({
                'Column': col,
                'Outliers': outlier_count,
                'Percentage': f"{outlier_pct:.1f}%",
                'Lower Bound': f"{lower_bound:.2f}",
                'Upper Bound': f"{upper_bound:.2f}",
                'Min': f"{self.df[col].min():.2f}",
                'Max': f"{self.df[col].max():.2f}"
            })
        
        summary_df = pd.DataFrame(outlier_summary)
        print(summary_df.to_string(index=False))
        print("\n")
        
        return summary_df
    
    def method1_winsorize(self, lower_percentile=0.05, upper_percentile=0.95):
        """
        Method 1: Winsorization
        Cap outliers at specified percentiles instead of removing them.
        """
        print(f"\nMETHOD 1: WINSORIZATION (capping at {lower_percentile*100}th and {upper_percentile*100}th percentiles)")
        print("-" * 70)
        
        df_winsorized = self.df.copy()
        
        for col in self.numerical_cols:
            lower_val = df_winsorized[col].quantile(lower_percentile)
            upper_val = df_winsorized[col].quantile(upper_percentile)
            
            original_min = df_winsorized[col].min()
            original_max = df_winsorized[col].max()
            
            df_winsorized[col] = df_winsorized[col].clip(lower=lower_val, upper=upper_val)
            
            print(f"{col}:")
            print(f"  Capped range: [{lower_val:.2f}, {upper_val:.2f}]")
            print(f"  Original range: [{original_min:.2f}, {original_max:.2f}]")
        
        return df_winsorized
    
    def method2_iqr_capping(self, multiplier=1.5):
        """
        Method 2: IQR-based Capping
        Cap outliers using IQR boundaries (Q1 - 1.5*IQR, Q3 + 1.5*IQR).
        """
        print(f"\nMETHOD 2: IQR-BASED CAPPING (multiplier={multiplier})")
        print("-" * 70)
        
        df_iqr = self.df.copy()
        
        for col in self.numerical_cols:
            Q1 = df_iqr[col].quantile(0.25)
            Q3 = df_iqr[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            capped_count = ((df_iqr[col] < lower_bound) | (df_iqr[col] > upper_bound)).sum()
            
            df_iqr[col] = df_iqr[col].clip(lower=lower_bound, upper=upper_bound)
            
            print(f"{col}: {capped_count} values capped to [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return df_iqr
    
    def method3_zscore_removal(self, threshold=3):
        """
        Method 3: Z-score based filtering
        Remove rows where any value has |z-score| > threshold.
        """
        print(f"\nMETHOD 3: Z-SCORE FILTERING (threshold={threshold})")
        print("-" * 70)
        
        df_zscore = self.df.copy()
        
        # Calculate z-scores for all numerical columns
        z_scores = np.abs(stats.zscore(df_zscore[self.numerical_cols]))
        
        # Keep rows where all z-scores are below threshold
        mask = (z_scores < threshold).all(axis=1)
        df_filtered = df_zscore[mask].copy()
        
        removed_count = len(df_zscore) - len(df_filtered)
        removed_pct = (removed_count / len(df_zscore)) * 100
        
        print(f"Rows removed: {removed_count} ({removed_pct:.1f}%)")
        print(f"Remaining rows: {len(df_filtered)}")
        
        return df_filtered
    
    def method4_log_transform(self):
        """
        Method 4: Log Transformation
        Apply log transformation to heavily skewed columns.
        """
        print("\nMETHOD 4: LOG TRANSFORMATION")
        print("-" * 70)
        
        df_log = self.df.copy()
        
        # Columns suitable for log transformation (positive values, right-skewed)
        log_candidates = ['IncomingCalls', 'AnsweredCalls', 'AbandonedCalls',
                         'ActualWorkloadMinutes', 'RequiredWorkloadMinutes']
        
        for col in log_candidates:
            if col in df_log.columns and df_log[col].min() > 0:
                skewness_before = df_log[col].skew()
                df_log[f'{col}_log'] = np.log1p(df_log[col])
                skewness_after = df_log[f'{col}_log'].skew()
                
                print(f"{col}:")
                print(f"  Skewness before: {skewness_before:.3f}")
                print(f"  Skewness after: {skewness_after:.3f}")
        
        return df_log
    
    def method5_robust_scaling(self):
        """
        Method 5: Median-based Outlier Capping
        Cap values using median ± k * MAD (Median Absolute Deviation).
        """
        print("\nMETHOD 5: ROBUST SCALING (Median ± MAD)")
        print("-" * 70)
        
        df_robust = self.df.copy()
        k = 3  # multiplier for MAD
        
        for col in self.numerical_cols:
            median = df_robust[col].median()
            mad = np.median(np.abs(df_robust[col] - median))
            
            lower_bound = median - k * mad
            upper_bound = median + k * mad
            
            capped_count = ((df_robust[col] < lower_bound) | (df_robust[col] > upper_bound)).sum()
            
            df_robust[col] = df_robust[col].clip(lower=lower_bound, upper=upper_bound)
            
            if capped_count > 0:
                print(f"{col}: {capped_count} values capped to [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return df_robust
    
    def compare_methods(self):
        """Compare statistics before and after each method."""
        print("\n" + "=" * 70)
        print("COMPARISON OF OUTLIER HANDLING METHODS")
        print("=" * 70)
        
        # Apply all methods
        df_winsor = self.method1_winsorize()
        df_iqr = self.method2_iqr_capping()
        df_zscore = self.method3_zscore_removal()
        df_log = self.method4_log_transform()
        df_robust = self.method5_robust_scaling()
        
        # Compare key metrics for a sample column
        sample_col = 'IncomingCalls'
        
        print(f"\nComparison for '{sample_col}':")
        print("-" * 70)
        
        methods = {
            'Original': self.df,
            'Winsorized': df_winsor,
            'IQR Capped': df_iqr,
            'Z-score Filtered': df_zscore,
            'Robust Scaled': df_robust
        }
        
        comparison = []
        for name, df in methods.items():
            comparison.append({
                'Method': name,
                'Count': len(df),
                'Mean': f"{df[sample_col].mean():.2f}",
                'Median': f"{df[sample_col].median():.2f}",
                'Std': f"{df[sample_col].std():.2f}",
                'Min': f"{df[sample_col].min():.2f}",
                'Max': f"{df[sample_col].max():.2f}"
            })
        
        comp_df = pd.DataFrame(comparison)
        print(comp_df.to_string(index=False))
        
        return {
            'winsorized': df_winsor,
            'iqr_capped': df_iqr,
            'zscore_filtered': df_zscore,
            'log_transformed': df_log,
            'robust_scaled': df_robust
        }
    
    def save_processed_data(self, processed_datasets):
        """Save all processed datasets to Excel files."""
        print("\n" + "=" * 70)
        print("SAVING PROCESSED DATASETS")
        print("=" * 70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for method_name, df in processed_datasets.items():
            filename = f"CallCenterData_{method_name}_{timestamp}.xlsx"
            df.to_excel(filename, index=False)
            print(f"Saved: {filename} ({len(df)} rows)")
        
        print("\nAll processed datasets saved successfully!")


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("CALL CENTER DATA - OUTLIER IMPACT REDUCTION")
    print("=" * 70)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Initialize handler
    handler = OutlierHandler('CallCenterData.xlsx')
    
    # Analyze outliers
    handler.analyze_outliers()
    
    # Compare all methods
    processed_datasets = handler.compare_methods()
    
    # Save processed data
    handler.save_processed_data(processed_datasets)
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
    1. WINSORIZATION: Best for preserving data size while limiting extreme values.
       - Use when you need all data points but want to reduce outlier influence
       - Recommended for time series analysis and forecasting
    
    2. IQR CAPPING: Similar to winsorization but uses quartile-based bounds.
       - Good for business metrics with known acceptable ranges
       - More aggressive than winsorization
    
    3. Z-SCORE FILTERING: Removes extreme outliers completely.
       - Use when outliers represent data errors or anomalies
       - Best for statistical modeling where assumptions matter
       - Note: Reduces dataset size
    
    4. LOG TRANSFORMATION: Reduces skewness in right-skewed distributions.
       - Best for heavily skewed call volume data
       - Makes data more normally distributed
       - Remember to back-transform for interpretation
    
    5. ROBUST SCALING: Uses median and MAD for outlier-resistant scaling.
       - Best when data has many outliers
       - Less sensitive to extreme values than mean/std
    
    RECOMMENDED APPROACH FOR CALL CENTER DATA:
    - For operational metrics: Use WINSORIZATION or IQR CAPPING
    - For predictive modeling: Use Z-SCORE FILTERING or LOG TRANSFORMATION
    - For visualization: Consider ROBUST SCALING or LOG TRANSFORMATION
    """)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
