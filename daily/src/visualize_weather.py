# visualize_weather.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import config

PLOTS_DIR = config.PLOT_DIR

def load_data(filepath):
    """Loads and performs basic cleaning on the weather data."""
    if not os.path.exists(filepath):
        print(f"Error: Data file not found at: {filepath}")
        return None
    
    print(f"Loading data from {filepath}...")
    df = pd.read_excel(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(by='datetime')
    print("Data loaded successfully.")
    return df

def get_season(month):
    """Determines the season based on the month."""
    # Rainy Season: May (5) to November (11)
    if 5 <= month <= 11:
        return 'Rainy'
    # Dry Season: December (12) to April (4)
    else:
        return 'Dry'

def preprocess_data(df):
    """Adds derived columns necessary for visualization."""
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['season'] = df['month'].apply(get_season)
    return df

def save_plot(fig, filename, dpi=120):
    """Saves the plot to the directory and closes it."""
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

# --- VISUALIZATION FUNCTIONS ---
def plot_daily_temp_with_confidence(df, window=30):
    """
    Visualizes the average daily temperature using a rolling mean (like an exponential moving average)
    and shows the standard deviation band, similar to the Training Reward chart.
    """
    print(f"Plotting: Daily temperature chart (Rolling Mean, window={window})...")

    # 1. Calculate Rolling Mean (smoother line)
    # The 'window' parameter defines the number of days to average over (e.g., 30 days = 1 month)
    df_temp = df[['datetime', 'temp']].set_index('datetime')
    rolling_mean = df_temp['temp'].rolling(window=window, center=True).mean()
    rolling_std = df_temp['temp'].rolling(window=window, center=True).std()

    # 2. Define the upper/lower bounds (Mean +/- Std Dev)
    upper_bound = rolling_mean + rolling_std
    lower_bound = rolling_mean - rolling_std

    # 3. Create the Plot
    fig, ax = plt.subplots(figsize=(14, 7))

    # Fill the area between the upper and lower bounds (similar to the shaded area in the original chart)
    ax.fill_between(df_temp.index, lower_bound, upper_bound,
                    color="#4e79a7", alpha=0.3, label="$\pm 1$ Std. Dev.")

    # Plot the Rolling Mean (the main, smooth line)
    ax.plot(df_temp.index, rolling_mean, color="#297fbd", linewidth=2.0, label=f"Avg. Temp ({window}-day Rolling Mean)")

    # Optional: Plot the raw daily temperature for context (very faint)
    # ax.plot(df_temp.index, df_temp['temp'], color='gray', linewidth=0.5, alpha=0.2, label="Daily Temp")


    ax.set_title(f"Average Daily Temperature in HCMC ({df['year'].min()}-{df['year'].max()} - Rolling Mean)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.5)

    # Clean up x-axis ticks to show years better
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())


    save_plot(fig, "daily_temp_timeseries.png")

def plot_monthly_temperature(df):
    """Visualizes the average monthly temperature."""
    print("Plotting: Average monthly temperature chart...")
    monthly_avg = df.groupby('month')['temp'].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(monthly_avg.index, monthly_avg.values, marker='o', color='orange', linewidth=2)
    ax.set_title('Average Monthly Temperature in HCMC (2015-2025)', fontsize=16, fontweight="bold")
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Avg Temperature (°C)', fontsize=12)
    ax.set_xticks(range(1, 13))
    ax.grid(True, linestyle='--', alpha=0.6)
    save_plot(fig, "monthly_avg_temp.png")

def plot_annual_temperature(df):
    """Visualizes the average annual temperature."""
    print("Plotting: Average annual temperature chart...")
    yearly_avg = df.groupby('year')['temp'].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(yearly_avg.index, yearly_avg.values, marker='o', color='blue', linewidth=2)
    ax.set_title('Average Annual Temperature in HCMC (2015-2025)', fontsize=16, fontweight="bold")
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Avg Temperature (°C)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    save_plot(fig, "annual_avg_temp.png")

def plot_seasonal_comparison(df):
    """Compares average temperature between seasons (Bar chart)."""
    print("Plotting: Seasonal temperature comparison chart...")
    season_avg = df.groupby('season')['temp'].mean().reindex(['Dry', 'Rainy']).reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=season_avg, x='season', y='temp', hue='season', palette='coolwarm', ax=ax, legend=False)
    ax.set_title('Average Temperature by Season (2015-2025)', fontsize=16, fontweight="bold")
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Avg Temperature (°C)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    save_plot(fig, "seasonal_temp_comparison.png")

def plot_correlation_heatmap(df):
    """Plots the correlation matrix."""
    print("Plotting: Correlation matrix...")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f", square=True, cbar_kws={"shrink": .8}, ax=ax, annot_kws={"size": 8})
    ax.set_title("Variable Correlation Matrix", fontsize=18, fontweight="bold")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    save_plot(fig, "correlation_heatmap.png", dpi=150)

def plot_temperature_distribution(df):
    """Visualizes the distribution of the average daily temperature."""
    print("Plotting: Temperature distribution histogram...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use seaborn's histplot for a better chart with KDE line
    sns.histplot(df['temp'], bins=30, kde=True, ax=ax, color='#3498db', edgecolor='white')
    
    # Get the KDE line from ax and set a darker color
    line = ax.lines[0]
    line.set_color('#2980b9')
    line.set_linewidth(2.5)

    ax.set_title('Distribution of Average Daily Temperature (2015-2025)', fontsize=16, fontweight="bold")
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Frequency (Days)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add a vertical line for the mean value
    mean_temp = df['temp'].mean()
    ax.axvline(mean_temp, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_temp:.2f}°C')
    ax.legend()
    
    save_plot(fig, "temp_distribution.png")

def plot_scatter_relationships(df):
    """Plots scatter plots showing relationships."""
    
    # 1. Temperature vs. Precipitation
    print("Plotting: Scatter Temperature vs. Precipitation...")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(data=df, x='temp', y='precip', 
                scatter_kws={'alpha': 0.3, 'color': 'steelblue', 's': 30},
                line_kws={'color': 'navy', 'linewidth': 2, 'linestyle': '--'}, 
                ax=ax)
    ax.set_title('Relationship between Temperature and Precipitation', fontsize=16, fontweight="bold")
    ax.set_xlabel('Avg Temperature (°C)')
    ax.set_ylabel('Precipitation (mm)')
    ax.grid(True, alpha=0.3)
    save_plot(fig, "scatter_temp_vs_precip.png")

    # 2. Temperature vs. Solar Radiation
    print("Plotting: Scatter Temperature vs. Solar Radiation...")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(data=df, x='solarradiation', y='temp', 
                scatter_kws={'alpha': 0.4, 'color': 'orange', 's': 30},
                line_kws={'color': 'darkorange', 'linewidth': 2, 'linestyle': '--'}, 
                ax=ax)
    ax.set_title('Relationship between Solar Radiation and Temperature', fontsize=16, fontweight="bold")
    ax.set_xlabel('Solar Radiation ($W/m^2$)', fontsize=12)
    ax.set_ylabel('Temperature (°C)')
    ax.grid(True, linestyle='--', alpha=0.5)
    save_plot(fig, "scatter_temp_vs_solar.png")

    # 3. Temperature vs. Humidity
    print("Plotting: Scatter Temperature vs. Humidity...")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(data=df, x='temp', y='humidity', scatter_kws={'alpha': 0.2, 'color': 'coral', 's': 30},
                line_kws={'color': 'firebrick', 'linewidth': 2, 'linestyle': '--'}, ax=ax)
    ax.set_title('Relationship between Temperature and Humidity', fontsize=16, fontweight="bold")
    ax.set_xlabel('Avg Temperature (°C)')
    ax.set_ylabel('Relative Humidity (%)')
    ax.grid(True, alpha=0.3)
    save_plot(fig, "scatter_temp_vs_humidity.png")
    
    # 4. Humidity vs. Precipitation
    print("Plotting: Scatter Humidity vs. Precipitation...")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x='humidity', y='precip', alpha=0.4, color='blue', ax=ax, s=30)
    ax.set_title('Relationship between Humidity and Precipitation', fontsize=16, fontweight="bold")
    ax.set_xlabel('Humidity (%)')
    ax.set_ylabel('Precipitation (mm)')
    ax.grid(True, linestyle='--', alpha=0.5)
    save_plot(fig, "scatter_humidity_vs_precip.png")

def plot_seasonal_boxplots(df):
    """Plots 3 boxplots comparing seasons."""
    print("Plotting: Boxplots comparing 2 seasons...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Temperature
    sns.boxplot(x='season', y='temp', data=df, hue='season', palette='coolwarm', ax=axes[0])
    axes[0].set_title("Temperature by Season", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Season")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.5)

    # Plot 2: Humidity
    sns.boxplot(x='season', y='humidity', data=df, hue='season', palette='coolwarm', ax=axes[1])
    axes[1].set_title("Humidity by Season", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Season")
    axes[1].set_ylabel("Humidity (%)")
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.5)

    # Plot 3: Precipitation
    sns.boxplot(x='season', y='precip', data=df, hue='season', palette='coolwarm', ax=axes[2])
    axes[2].set_title("Precipitation by Season", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Season")
    axes[2].set_ylabel("Precipitation (mm)")
    axes[2].grid(True, axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_plot(fig, "seasonal_boxplots.png")

def plot_climographs(monthly_data):
    """Plots the climographs."""
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # 1. Climograph: Temperature and Precipitation
    print("Plotting: Climograph (Temperature & Precipitation)...")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color_temp = '#d9886a'
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Avg Temperature (°C)', color=color_temp, fontsize=12)
    ax1.plot(monthly_data['month'], monthly_data['temp'], color=color_temp, marker='o', linewidth=2.5, label='Temperature (°C)')
    ax1.tick_params(axis='y', labelcolor=color_temp)
    ax1.set_ylim(24, 32)
    
    ax2 = ax1.twinx()
    color_precip = '#9b6fcc'
    ax2.set_ylabel('Avg Precipitation (mm)', color=color_precip, fontsize=12)
    ax2.bar(monthly_data['month'], monthly_data['precip'], color=color_precip, alpha=0.7, width=0.6, label='Precipitation (mm)')
    ax2.tick_params(axis='y', labelcolor=color_precip)
    
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(month_names, fontsize=11)
    ax1.set_title('Average Monthly Temperature and Precipitation (2015-2025)', fontsize=16, fontweight='bold')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    save_plot(fig, "climograph_temp_precip.png")

    # 2. Climograph: Temperature and Humidity
    print("Plotting: Climograph (Temperature & Humidity)...")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color_temp = '#e07a5f'
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Avg Temperature (°C)', color=color_temp, fontsize=12)
    ax1.plot(monthly_data['month'], monthly_data['temp'], color=color_temp, marker='o', linewidth=2.5, label='Temperature (°C)')
    ax1.tick_params(axis='y', labelcolor=color_temp)
    ax1.set_ylim(24, 32)
    
    ax2 = ax1.twinx()
    color_humidity = '#4a90e2'
    ax2.set_ylabel('Avg Humidity (%)', color=color_humidity, fontsize=12)
    ax2.bar(monthly_data['month'], monthly_data['humidity'], color=color_humidity, alpha=0.75, width=0.55, label='Humidity (%)')
    ax2.tick_params(axis='y', labelcolor=color_humidity)
    ax2.set_ylim(0, 100)
    
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(month_names, fontsize=11)
    ax1.set_title('Average Monthly Temperature and Humidity (2015-2025)', fontsize=16, fontweight='bold')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    save_plot(fig, "climograph_temp_humidity.png")

# --- MAIN FUNCTION TO RUN ---

def main():
    """Main function to coordinate data loading and plotting."""
    
    # Create plots directory if it doesn't exist
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        print(f"Created directory: {PLOTS_DIR}")

    # 1. Get raw file path from config
    raw_path = os.path.join(config.RAW_DATA_DIR, config.RAW_FILE_NAME)
    
    # 2. CALL load_data AND ASSIGN THE RESULT TO 'df'
    #    This is the step that creates the 'df' variable
    df = load_data(raw_path) 
    
    # 3. Check if load_data was successful
    if df is None:
        print("❌ Stopping program because data could not be loaded.")
        return # Exit the main function

    # 4. Now the 'df' variable exists and is safe to use
    df = preprocess_data(df)
    
    # Calculate monthly aggregate data (for climographs)
    monthly_data = df.groupby('month').agg({
        'temp': 'mean',
        'precip': 'mean',
        'humidity': 'mean'
    }).reset_index()
    
    # Run the visualization functions
    plot_daily_temp_with_confidence(df, window=30) # 30 ngày tương đương với 1 tháng    plot_monthly_temperature(df)
    plot_annual_temperature(df)
    plot_seasonal_comparison(df)
    plot_temperature_distribution(df)
    plot_correlation_heatmap(df)
    plot_scatter_relationships(df)
    plot_seasonal_boxplots(df)
    plot_climographs(monthly_data)
    
    print("\n--- COMPLETE! ---")
    print(f"All plots have been saved to the directory: '{PLOTS_DIR}'")

if __name__ == "__main__":
    main()