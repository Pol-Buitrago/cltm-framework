import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d

# --- 1. Configuration and Parameter Definitions ---
INPUT_DIR = "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/subsets/gender"
OUTPUT_PLOTS_DIR = "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/scripts/interval/plots"
FILE_PATTERN = INPUT_DIR + "/*.f1_by_samples.csv"

# Operational Criterion Thresholds (Section 6 Definitions)
TAU = 0.05       # tau: Minimum relative gain
Z_SNR = 1.3      # z: SNR threshold in SE units (90% 1 cola, 80% dos colas)
EPSILON = 0.05   # epsilon: Minimum slope per log(N) unit
ALPHA = 2        # Data increase factor (alpha=2 for [N, 2N])
LOG_ALPHA = np.log(ALPHA) # l = log(alpha)

# Smoothing parameter for the Gaussian filter
SMOOTHING_SIGMA = 5 

# Outlier removal parameter (1.5 * IQR is standard)
OUTLIER_FACTOR_IQR = 1.5 

# Ensure the output directory exists
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
print(f"Plot directory created/verified: {OUTPUT_PLOTS_DIR}")

# --- 2. CI_min Calculation Function ---
def calculate_ci_min(perf_n, std_n, perf_2n, std_2n):
    """
    Calculates the Minimum Confidence Index (CI_min) for the interval [N, 2N].
    """
    delta = perf_2n - perf_n
    
    if delta <= 0:
        return 0.0

    se_delta = np.sqrt(std_n**2 + std_2n**2)
    
    ci_rel_gain = delta / (TAU * perf_n)
    ci_snr = delta / (Z_SNR * se_delta) if se_delta != 0 else np.inf
    ci_slope = (delta / LOG_ALPHA) / EPSILON
    
    return min(ci_rel_gain, ci_snr, ci_slope)

# --- 3. Data Processing and CI_min Calculation ---
all_results = []

print(f"\nSearching for files in: {INPUT_DIR}")
for filepath in glob.glob(FILE_PATTERN):
    lang_code = filepath.split('/')[-1].split('.')[0]
    
    try:
        df = pd.read_csv(filepath)
        df = df.sort_values(by='num_samples').reset_index(drop=True)
        
        for idx in df.index:
            N = df.loc[idx, 'num_samples']
            target_2N = N * ALPHA
            
            df_2N = df[df['num_samples'] == target_2N]

            if not df_2N.empty:
                perf_n = df.loc[idx, 'f1']
                std_n = df.loc[idx, 'f1_std']
                perf_2n = df_2N['f1'].iloc[0]
                std_2n = df_2N['f1_std'].iloc[0]
                
                ci_min = calculate_ci_min(perf_n, std_n, perf_2n, std_2n)
                
                all_results.append({
                    'lang': lang_code,
                    'N': N,
                    '2N': target_2N,
                    'CI_min': ci_min,
                })
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

results_df = pd.DataFrame(all_results)

# --- 4. Global CI_min Calculation and Outlier Removal ---

# Aggregate by N and calculate the global median CI_min
global_ci_min_df = results_df.groupby('N').agg(
    median_CI_min=('CI_min', 'median'),
    count_langs=('lang', 'count')
).reset_index()
global_ci_min_df['2N'] = global_ci_min_df['N'] * ALPHA

# --- 4.1. Outlier Removal (using IQR method on the median CI_min values) ---
global_ci_min_filtered_df = global_ci_min_df.copy()
outliers_df = pd.DataFrame()

if not global_ci_min_df.empty:
    Q1 = global_ci_min_df['median_CI_min'].quantile(0.25)
    Q3 = global_ci_min_df['median_CI_min'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define the upper boundary for outliers
    upper_bound = Q3 + OUTLIER_FACTOR_IQR * IQR
    
    # Filter the DataFrame to remove points above the upper bound
    global_ci_min_filtered_df = global_ci_min_df[global_ci_min_df['median_CI_min'] <= upper_bound].copy()
    
    # Store outliers separately for reporting
    outliers_df = global_ci_min_df[global_ci_min_df['median_CI_min'] > upper_bound].copy()
    
    if not outliers_df.empty:
        print(f"\nOutlier Filtering: IQR={IQR:.3f}, Upper Bound={upper_bound:.3f}.")
        print(f"Removed {len(outliers_df)} outlier points from smoothing/plotting calculations.")
        print(outliers_df[['N', 'median_CI_min']].to_markdown(index=False, floatfmt=".3f", headers=['N', 'Outlier CI_min']))
    else:
        print("\nNo outliers detected for filtering.")
else:
    print("No data available for outlier filtering.")

# Identify global dynamic intervals (using the UNFILTERED data for report consistency in table)
global_dynamic_intervals = global_ci_min_df[global_ci_min_df['median_CI_min'] >= 1.0].copy()

# --- 4.2. Gaussian Smoothing for Robust N_opt (uses FILTERED data) ---
best_N_robust = None
log_N_uniform = np.array([])
CI_smoothed = np.array([])

if not global_ci_min_filtered_df.empty:
    
    global_ci_min_filtered_df = global_ci_min_filtered_df.sort_values(by='N').reset_index(drop=True)
    
    log_N = np.log10(global_ci_min_filtered_df['N'].values)
    median_CI = global_ci_min_filtered_df['median_CI_min'].values

    # 1. Create a uniformly spaced X-axis in log-scale
    log_N_uniform = np.linspace(log_N.min(), log_N.max(), 200)
    
    # 2. Interpolate CI_min values onto the uniform axis
    CI_interpolated = np.interp(log_N_uniform, log_N, median_CI)
    
    # 3. Apply Gaussian Smoothing
    CI_smoothed = gaussian_filter1d(CI_interpolated, sigma=SMOOTHING_SIGMA)
    
    # 4. Find the maximum of the smoothed curve (Robust Optimum)
    max_index_smoothed = np.argmax(CI_smoothed)
    N_opt_smoothed = 10**(log_N_uniform[max_index_smoothed])
    
    # 5. Find the closest discrete N to the robust optimum (uses the UNFILTERED data for final N for consistency)
    idx_nearest = np.argmin(np.abs(global_ci_min_df['N'].values - N_opt_smoothed))
    best_N_robust = global_ci_min_df.loc[idx_nearest]
    
# --- 5. Tabular Results Presentation ---

print("\n" + "="*80)
print("🎯 GLOBAL DYNAMIC REGIME ANALYSIS")
print(f"Threshold CI_min: 1.0 | Parameters: tau={TAU}, z={Z_SNR}, epsilon={EPSILON}")
print("="*80)

if best_N_robust is not None:
    print("⭐ ROBUST OPTIMAL INTERVAL (Based on Gaussian Smoothing):")
    print(f"N_opt: {int(best_N_robust['N'])} | Interval: [{int(best_N_robust['N'])}, {int(best_N_robust['2N'])}]")
    print(f"Median CI_min (at this discrete point): {best_N_robust['median_CI_min']:.3f}\n")
    
    print("✅ VALID GLOBAL [N, 2N] INTERVALS (Median CI_min >= 1.0):")
    print(global_dynamic_intervals[['N', '2N', 'median_CI_min', 'count_langs']].to_markdown(
        index=False, 
        floatfmt=".3f", 
        headers=['N', '2N', 'Median CI_min', 'Num Languages']
    ))
    
else:
    print("❌ No data available or no global intervals found where Median CI_min >= 1.0.")

print("="*80)


# --- 6. Improved Plot Generation (Smoothed Curve with Outliers REMOVED from plot) ---

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')

fig, ax = plt.subplots(figsize=(12, 7))

# 1. Filtered Discrete Data Points (Used for smoothing and main plot scale)
# *** NOTE: Outliers are not plotted here, ensuring a clean Y-axis scale ***
ax.plot(global_ci_min_filtered_df['N'], global_ci_min_filtered_df['median_CI_min'], 
        'o', color='gray', alpha=0.5, markersize=5, label='Filtered Discrete Median CI_min')


# 2. Smoothed Curve (Robust Trend)
if not global_ci_min_filtered_df.empty: # Only plot if there's data after filtering
    ax.plot(10**log_N_uniform, CI_smoothed, 
            '-', color='#d62728', linewidth=2.5, 
            label=f'Smoothed Trend (Gaussian $\sigma={SMOOTHING_SIGMA}$)')

# 3. Dynamic Threshold Line
ax.axhline(1.0, color='b', linestyle='--', linewidth=1.5, label='Dynamic Threshold (1.0)')

# 4. Highlight the Optimal Robust N
if best_N_robust is not None:
    N_opt_disc = best_N_robust['N']
    CI_opt_disc = best_N_robust['median_CI_min']
    
    # Mark the robust optimal discrete point 
    ax.plot(N_opt_disc, CI_opt_disc, 'X', color='green', markersize=12, 
            label=f'Robust Optimal N: {int(N_opt_disc)}', zorder=5)
    
    # Annotation for the Optimal Interval
    # We use a slight margin (1.05) above the max filtered value for the upper y-limit for safety.
    max_y_filtered = global_ci_min_filtered_df['median_CI_min'].max()


    # Highlight the region where CI_min >= 1.0 on the smoothed curve
    if not global_ci_min_filtered_df.empty:
        valid_range = CI_smoothed >= 1.0
        ax.fill_between(
            10**log_N_uniform, 
            1.0, 
            CI_smoothed, 
            where=valid_range, 
            color='gold', 
            alpha=0.3,
            label='Global Dynamic Regime'
        )
    
# Configuration for clarity
ax.set_xscale('log')
ax.set_xlabel('Initial Sample Size $N$ (Log Scale)', fontsize=12)
ax.set_ylabel(r'Global Median of $\mathrm{CI}_{\min}$', fontsize=12)
ax.set_title(r'Robust Identification of the Global Dynamic Regime (Outliers Excluded)', fontsize=14)
ax.legend(loc='best')
ax.grid(True, which="both", ls="--", alpha=0.5)

plot_filename_smoothed = os.path.join(OUTPUT_PLOTS_DIR, "03_global_median_ci_min_smoothed_filtered_final.png")
fig.savefig(plot_filename_smoothed, bbox_inches='tight')
plt.close(fig)

print(f"\nPlot saved to: {plot_filename_smoothed}")