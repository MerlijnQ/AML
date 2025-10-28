import os
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df_orig = pd.read_csv('dataset/continuous dataset.csv')

# Ensure datetime is parsed correctly until covid happened
df_orig['datetime'] = pd.to_datetime(df_orig['datetime'])
df_orig = df_orig.sort_values('datetime')
covid_start_date = pd.to_datetime('2020-03-01')
df = df_orig[df_orig['datetime'] <= covid_start_date].copy()
df['day_of_week'] = df['datetime'].dt.day_of_week

# Hour of the day
df['hour'] = df['datetime'].dt.hour

# Minute of the day
df['minute_of_day'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute

# Holidays (ID 0-22)
unique_holiday_IDs = df['Holiday_ID'].unique()

def save_fig(title):
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/{title}.png")

# Plot to see when covid happened
def plot_covid():
    start_date = pd.to_datetime('2020-03-01')
    end_date = pd.to_datetime('2020-04-01')

    # Filter dataframe for the selected date range
    df_covid = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]

    plt.figure(figsize=(12, 6))
    plt.plot(df_covid['datetime'], df_covid['nat_demand'], label='Original Data', alpha=0.5)
    plt.xlabel('Date')
    plt.ylabel('Nat Demand')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_grid_load():
    global df
    rel_directory = "images/seasonal_decomposition/"
    window_size_week = 24 * 7  # a week
    window_size_4_weeks = 24 * 7 * 4  # 4 weeks ~ 1 month
    covid_start_date = pd.to_datetime('2020-03-01')

    # trend
    df['monthly_trend'] = df['nat_demand'].rolling(window=window_size_4_weeks, center=True).mean()
    df['seasonality'] = (df['nat_demand'] - df['monthly_trend']).rolling(window=window_size_4_weeks, center=True).mean()
    df['residual'] = df['nat_demand'] - df['monthly_trend'] - df['seasonality']

    os.makedirs("images/seasonal_decomposition", exist_ok=True)
    start_date = df['datetime'].min()
    end_date = df['datetime'].max()

    def plot_figure(y_values, figurename, label = "", y_label="Grid load [MWh]", color='blue', linewidth=1):
        # --- Plot 1: Full grid load ---
        plt.figure(figsize=(12, 4))
        plt.plot(df['datetime'], y_values, label=label, alpha=0.5, color=color, linewidth=linewidth)
        plt.xlabel("Date")
        plt.ylabel(y_label)
        plt.legend(loc='lower left')
        # plt.grid(True)
        plt.xlim(start_date, end_date)
        plt.ylim(-400, 1750)
        plt.tight_layout()
        plt.savefig(rel_directory + figurename + ".pdf")
        plt.close()

    # --- Plot 1: Full grid load ---
    plot_figure(df['nat_demand'], "full_grid_load", "Original Data")

    # --- Plot 2: Trend ---
    plot_figure(df['monthly_trend'], "trend", label="Trend (4-Weekly MA)", color='green', linewidth=3)

    # --- Plot 3: Seasonality ---
    plot_figure(df['seasonality'], "seasonality", label='Seasonality', color='green', linewidth=3)

    # --- Plot 4: Residual ---
    plot_figure(df['residual'], "residual", label='Residual', color='green', linewidth=1)

def plot_grid_load_intra_day():
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True)
    avg_daily = df.groupby('day_of_week')['nat_demand'].mean()
    avg_hourly = df.groupby('hour')['nat_demand'].mean()
    avg_minutely = df.groupby('minute_of_day')['nat_demand'].mean()

    # Plot 1: Weekly (based on hours)
    axes[0].plot(avg_daily.index, avg_daily.values, marker='o')
    axes[0].set_title('Average Grid Load per Day of the Week')
    axes[0].set_xlabel('Day of week')
    axes[0].set_ylabel('Grid Load [MWh]')
    axes[0].grid(True)

    # Plot 2: Daily (based on minutes)
    axes[1].plot(avg_hourly.index, avg_hourly.values, marker='o')
    axes[1].set_title('Average Daily Grid Load by Hour')
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Grid Load [MWh]')
    axes[1].grid(True)

    plt.tight_layout()
    save_fig('weekly_daily_average')
    plt.show()

def plot_holiday_IDs():
    avg_demand = df['nat_demand'].mean()

    # Compute mean and standard deviation per Holiday_ID
    grouped = df.groupby('Holiday_ID')['nat_demand']
    avg_holiday = grouped.mean()
    std_holiday = grouped.std()

    plt.figure(figsize=(10, 5))
    # Plot with error bars representing standard deviation
    plt.errorbar(
        avg_holiday.index,
        avg_holiday.values,
        yerr=std_holiday.values,
        fmt='o-',
        ecolor='gray',
        elinewidth=1,
        capsize=4,
        label='Deviation ± 1 SD'
    )

    # Add average demand line
    plt.axhline(
        y=avg_demand,
        color='red',
        linestyle='--',
        linewidth=1.5,
        label='Average Demand'
    )

    plt.title('Average Daily Grid Load per Holiday')
    plt.xlabel('Holiday ID')
    plt.ylabel('Grid Load [MWh]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_fig('holiday_grid_loads')
    plt.show()

def plot_full_grid_load():
    # --- Plot 1: Full grid load ---
    plt.figure(figsize=(12, 4))
    plt.plot(df_orig['datetime'], df_orig["nat_demand"], label="nat demand", alpha=0.5)
    plt.xlabel("Date")
    plt.ylabel("National Demand [MWh]")
    # plt.legend(loc='lower left')
    # plt.grid(True)
    plt.xlim(df_orig['datetime'].min(), df_orig['datetime'].max())
    plt.ylim(-400, 1750)
    plt.tight_layout()
    plt.savefig('images/' + "full_grid_load" + ".pdf")
    plt.show()
    plt.close()


# plot_grid_load()
# plot_grid_load_intra_day()
# plot_holiday_IDs()
# plot_full_grid_load()
print(df_orig['datetime'].max())