import os
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('dataset/continuous dataset.csv')

# Ensure datetime is parsed correctly
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')

df['day_of_week'] = df['datetime'].dt.day_of_week

# Hour of the day
df['hour'] = df['datetime'].dt.hour

# Minute of the day
df['minute_of_day'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute



# Plot to see when covid happened
def plot_covid():
    start_date = pd.to_datetime('2020-03-01')
    end_date = pd.to_datetime('2020-04-01')

    # Filter dataframe for the selected date range
    df_covid = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]

    plt.figure(figsize=(12, 6))
    plt.plot(df_covid['datetime'], df_covid['nat_demand'], label='Original Data', alpha=0.5)
    plt.title(f'National Grid Load of Panama ({start_date} to {end_date})')
    plt.xlabel('Date')
    plt.ylabel('Nat Demand')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_grid_load():
    global df
    # Apply moving average to extract trend
    window_size_week = 24 * 7  # a week
    window_size_4_weeks = 24 * 7 * 4  # 4 weeks ~ 1 month
    covid_start_date = pd.to_datetime('2020-03-01')

    # trend
    df['monthly_trend'] = df['nat_demand'].rolling(window=window_size_4_weeks, center=True).mean()
    # avg_demand = df['nat_demand'].mean()
    # seasonality
    df['seasonality'] = (df['nat_demand'] - df['monthly_trend']).rolling(window=window_size_4_weeks, center=True).mean()
    # residual = full_load - trend - seasonality
    df['residual'] = df['nat_demand'] - df['monthly_trend'] - df['seasonality']
    

    # Create two vertically stacked subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    # --- Plot 1: Full range ---
    axes[0].plot(df['datetime'], df['nat_demand'], label='Original Data', alpha=0.5)
    axes[0].set_title('National Grid Load of Panama')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Grid load [MWh]')
    axes[0].legend()
    axes[0].grid(True)


    df = df[(df['datetime'] <= covid_start_date)]
    # --- Plot 2: Zoomed in on y-axis and x-axis until covid ---
    # axes[1].plot(df['datetime'], df['nat_demand'], label='Original Data', alpha=0.5)
    axes[1].plot(df['datetime'], df['monthly_trend'], label='Trend (4-Weekly Moving Average)', color='green', linewidth=3)
    axes[1].set_ylim(900, 1400)  # zoom in above 750
    axes[1].set_title('Trend')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Grid load [MWh]')
    axes[1].legend()
    axes[1].grid(True)

    # --- Plot 3: Seasonality
    # axes[2].plot(df['datetime'], df['nat_demand'], label='Original Data', alpha=0.5)
    axes[2].plot(df['datetime'], df['seasonality'], label='seasonality', color='green', linewidth=3)
    # axes[2].set_ylim(900, 1400)  # zoom in above 750
    axes[2].set_title('Seasonality')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Grid load [MWh]')
    axes[2].legend()
    axes[2].grid(True)

    # --- Plot 3: Residual    
    axes[3].plot(df['datetime'], df['residual'], label='residual', color='green', linewidth=1)
    # axes[2].set_ylim(900, 1400)  # zoom in above 750
    axes[3].set_title('Residual load')
    axes[3].set_xlabel('Date')
    axes[3].set_ylabel('Grid load [MWh]')
    axes[3].legend()
    axes[3].grid(True)
    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/seasonal_decomposition.png")
    plt.show()

def plot_grid_load_intra_day():
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True)
    avg_daily = df.groupby('day_of_week')['nat_demand'].mean()
    avg_hourly = df.groupby('hour')['nat_demand'].mean()
    avg_minutely = df.groupby('minute_of_day')['nat_demand'].mean()

    # Plot 1: Weekly (based on hours)
    axes[0].plot(avg_daily.index, avg_daily.values, marker='o')
    axes[0].set_title('Average Grid Load per Day of the Week')
    axes[0].set_xlabel('Day of week')
    axes[0].set_ylabel('Average Grid Load')
    axes[0].grid(True)

    # Plot 2: Daily (based on minutes)
    axes[1].plot(avg_hourly.index, avg_hourly.values, marker='o')
    axes[1].set_title('Average Daily Grid Load by Hour')
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Average Grid Load')
    axes[1].grid(True)

    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/weekly_daily_average.png")
    plt.show()

plot_grid_load_intra_day()