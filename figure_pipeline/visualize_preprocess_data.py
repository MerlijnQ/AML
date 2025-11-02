import os
import pandas as pd
import matplotlib.pyplot as plt

class DatasetVisualizer:
    def __init__(self):
        # read dataset
        self.df_orig = self.__get_modified_dataset()

        # get pre-covid dataset
        self.covid_start_date = pd.to_datetime('2020-03-01')
        self.df_pre_covid = self.df_orig[self.df_orig['datetime'] <= self.covid_start_date].copy()

        self.start_date_dataset = self.df_orig['datetime'].min()
        self.end_date_dataset = self.df_orig['datetime'].max()

        # Holidays (ID 0-22)
        unique_holiday_IDs = self.df_orig['Holiday_ID'].unique()

        self.rel_dir = 'images/data_preprocessing/'

        self.fontsize = 22
        self.label_size = 18

    def __get_modified_dataset(self):
        # read dataset
        rel_path = '../dataset/continuous dataset.csv'
        df = pd.read_csv(rel_path)

        # convert time in dataset to a pandas time and sort it
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')

        # add time variables
        df['day_of_week'] = df['datetime'].dt.day_of_week
        df['hour'] = df['datetime'].dt.hour
        df['minute_of_day'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute

        return df

    def __make_fig(self):
        plt.figure(figsize=(12, 4))

    def __plot_seasonlity_figure(self, y_values, figurename, label = "", y_label="Grid load [MWh]", color='blue', linewidth=1):
        plt.figure(figsize=(12, 4))
        plt.plot(self.df_pre_covid['datetime'], y_values, label=label, alpha=0.5, color=color, linewidth=linewidth)
        plt.xlabel("Date", fontsize=self.fontsize)
        plt.ylabel(y_label, fontsize=self.fontsize)
        plt.xlim(self.start_date_dataset, self.covid_start_date)
        plt.ylim(-400, 1750)
        self.__set_x_and_y_ticks()
        plt.tight_layout()
        self.__save_fig(figurename)
        plt.show()
        plt.close()

    def __save_fig(self, title):
        os.makedirs(self.rel_dir, exist_ok=True)
        plt.savefig(f"{self.rel_dir}/{title}.pdf")

    def __set_x_and_y_ticks(self):
        plt.xticks(fontsize=self.label_size)  # x-axis tick labels
        plt.yticks(fontsize=self.label_size)  # y-axis tick labels
    # Plot to see when covid happened
    def plot_covid(self, show=False):
        start_date = pd.to_datetime('2020-03-01')
        end_date = pd.to_datetime('2020-04-01')

        # Filter dataframe for the selected date range
        df_covid = self.df_orig[(self.df_orig['datetime'] >= start_date) & (self.df_orig['datetime'] <= end_date)]

        self.__make_fig()
        plt.plot(df_covid['datetime'], df_covid['nat_demand'], label='Original Data', alpha=0.5)
        plt.xlabel('Date', fontsize=self.fontsize)
        plt.ylabel('Nat Demand', fontsize=self.fontsize)
        plt.xlim(start_date, end_date)
        plt.ylim(0, 1700)
        plt.legend()
        plt.tight_layout()
        self.__save_fig('march_2020')
        if show:
            plt.show()

    def plot_full_grid_load(self):
        # --- Plot 1: Full grid load ---
        self.__make_fig()
        plt.plot(self.df_orig['datetime'], self.df_orig["nat_demand"], label="nat demand", alpha=0.5)
        plt.xlabel("Date", fontsize=self.fontsize)
        plt.ylabel("National Demand [MWh]", fontsize=self.fontsize)
        # plt.legend(loc='lower left')
        plt.xlim(self.df_orig['datetime'].min(), self.df_orig['datetime'].max())
        plt.ylim(0, 1750)
        self.__set_x_and_y_ticks()
        plt.tight_layout()
        self.__save_fig('full_grid_load')
        plt.show()
        plt.close()

    def plot_grid_load_intra_day(self):
        avg_daily = self.df_pre_covid.groupby('day_of_week')['nat_demand'].mean()
        avg_hourly = self.df_pre_covid.groupby('hour')['nat_demand'].mean()

        # Figure 1: Weekly average
        self.__make_fig()
        plt.plot(avg_daily.index, avg_daily.values, marker='o')
        plt.xlabel('Day of Week', fontsize=self.fontsize)
        plt.ylabel('Grid Load [MWh]', fontsize=self.fontsize)
        plt.ylim(900, 1400)
        plt.xlim(0, 6)
        self.__set_x_and_y_ticks()
        plt.tight_layout()
        self.__save_fig('weekly_average')
        plt.show()

        # Figure 2: Daily hourly average
        plt.figure(figsize=(12, 4))
        plt.plot(avg_hourly.index, avg_hourly.values, marker='o')
        plt.xlabel('Hour of Day', fontsize=self.fontsize)
        plt.ylabel('Grid Load [MWh]', fontsize=self.fontsize)
        plt.ylim(800, 1500)
        plt.xlim(0, 23)
        self.__set_x_and_y_ticks()
        plt.tight_layout()
        self.__save_fig('daily_average')
        plt.show()

    def plot_grid_load_full_dataset(self):
        plt.figure(figsize=(12, 4))
        plt.plot(self.df_orig['datetime'], self.df_orig['nat_demand'], alpha=0.5, linewidth=0.5)
        plt.xlabel("Date", fontsize=self.fontsize)
        plt.ylabel('nat_demand [MWh]', fontsize=self.fontsize)
        plt.xlim(self.start_date_dataset, self.end_date_dataset)
        plt.ylim(-400, 1750)
        self.__set_x_and_y_ticks()
        plt.tight_layout()
        self.__save_fig('grid_load_full_dataset')
        plt.show()
        plt.close()

    def plot_grid_load(self):
        window_size_4_weeks = 24 * 7 * 4  # 4 weeks ~ 1 month

        # calculate the trend, seasonlity and residual data
        self.df_pre_covid['monthly_trend'] = self.df_pre_covid['nat_demand'].rolling(window=window_size_4_weeks, center=True).mean()
        self.df_pre_covid['seasonality'] = (self.df_pre_covid['nat_demand'] - self.df_pre_covid['monthly_trend']).rolling(window=window_size_4_weeks, center=True).mean()
        self.df_pre_covid['residual'] = self.df_pre_covid['nat_demand'] - self.df_pre_covid['monthly_trend'] - self.df_pre_covid['seasonality']

        os.makedirs("images/seasonal_decomposition", exist_ok=True)
        start_date = self.df_pre_covid['datetime'].min()
        end_date = self.df_pre_covid['datetime'].max()

        # --- Plot 1: Full grid load ---
        self.__plot_seasonlity_figure(self.df_pre_covid['nat_demand'], "full_grid_load", "Original Data")

        # --- Plot 2: Trend ---
        self.__plot_seasonlity_figure(self.df_pre_covid['monthly_trend'], "trend", label="Trend (4-Weekly MA)", color='green', linewidth=3)

        # --- Plot 3: Seasonality ---
        self.__plot_seasonlity_figure(self.df_pre_covid['seasonality'], "seasonality", label='Seasonality', color='green', linewidth=3)

        # --- Plot 4: Residual ---
        self.__plot_seasonlity_figure(self.df_pre_covid['residual'], "residual", label='Residual', color='green', linewidth=1)

    def plot_holiday_IDs(self):
        avg_demand = self.df_pre_covid['nat_demand'].mean()

        # Compute mean and standard deviation per Holiday_ID
        grouped = self.df_pre_covid.groupby('Holiday_ID')['nat_demand']
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

        plt.xlabel('Holiday ID', fontsize=self.fontsize)
        plt.ylabel('Grid Load [MWh]', fontsize=self.fontsize)
        self.__set_x_and_y_ticks()
        plt.tight_layout()
        self.__save_fig('holiday_grid_loads')
        plt.show()

    
if __name__ == "__main__":
    datavisualizer = DatasetVisualizer()
    datavisualizer.plot_full_grid_load()
    datavisualizer.plot_grid_load_intra_day()
    datavisualizer.plot_grid_load()
    # datavisualizer.plot_covid()
    datavisualizer.plot_holiday_IDs()
    datavisualizer.plot_grid_load_full_dataset()
