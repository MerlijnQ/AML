import os
from typing import List, Optional, Union
import pandas as pd
import matplotlib.pyplot as plt

class DatasetVisualizer:
    """
    A class for visualizing the grid load dataset
    """

    def __init__(self) -> None:
        """
        Initializes the DatasetVisualizer by reading and preprocessing
        the dataset, and setting plotting defaults.
        """
        # Read dataset
        self.df_orig: pd.DataFrame = self.__get_modified_dataset()

        # Get pre-COVID dataset
        self.covid_start_date: pd.Timestamp = pd.to_datetime('2020-03-01')
        self.df_pre_covid: pd.DataFrame = self.df_orig[self.df_orig['datetime'] <= self.covid_start_date].copy()

        self.start_date_dataset: pd.Timestamp = self.df_orig['datetime'].min()
        self.end_date_dataset: pd.Timestamp = self.df_orig['datetime'].max()

        # Holidays (ID 0-22)
        unique_holiday_IDs: List[int] = self.df_orig['Holiday_ID'].unique().tolist()

        # Plotting parameters
        self.rel_dir: str = 'images/data_preprocessing/'
        self.fontsize: int = 22
        self.label_size: int = 18
        self.time_window_names: List[str] = []  # Placeholder if needed for RMSE plots

    def __get_modified_dataset(self) -> pd.DataFrame:
        """
        Reads the dataset, converts datetime, sorts it,
        and adds additional time features.

        Returns:
            pd.DataFrame: The modified dataset so it is usefull for plotting.
        """
        rel_path: str = '../dataset/continuous dataset.csv'
        df: pd.DataFrame = pd.read_csv(rel_path)

        # Convert time in dataset to pandas datetime and sort
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')

        # Add time variables
        df['day_of_week'] = df['datetime'].dt.day_of_week
        df['hour'] = df['datetime'].dt.hour
        df['minute_of_day'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute

        return df

    def __make_fig(self) -> None:
        """Creates a new matplotlib figure with predefined size."""
        plt.figure(figsize=(12, 4))

    def __plot_seasonlity_figure(
        self,
        y_values: Union[pd.Series, List[float]],
        figurename: str,
        label: str = "",
        y_label: str = "Grid load [MWh]",
        color: str = 'blue',
        linewidth: int = 1
    ) -> None:
        """
        Plots seasonality-related figures including trend, residual, etc.

        Args:
            y_values (Union[pd.Series, List[float]]): Values to plot.
            figurename (str): Filename for saving figure.
            label (str, optional): Line label. Defaults to "".
            y_label (str, optional): Y-axis label. Defaults to "Grid load [MWh]".
            color (str, optional): Line color. Defaults to 'blue'.
            linewidth (int, optional): Line width. Defaults to 1.
        """
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

    def __save_fig(self, title: str) -> None:
        """
        Saves the current matplotlib figure as a PDF.

        Args:
            title (str): File name to save.
        """
        os.makedirs(self.rel_dir, exist_ok=True)
        plt.savefig(f"{self.rel_dir}/{title}.pdf")

    def __set_x_and_y_ticks(self) -> None:
        """Sets x-axis and y-axis tick label font sizes."""
        plt.xticks(fontsize=self.label_size)
        plt.yticks(fontsize=self.label_size)

    def plot_covid(self, show: bool = False) -> None:
        """
        Plots national demand during the first month of COVID.

        Args:
            show (bool, optional): Whether to display the plot. Defaults to False.
        """
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

    def plot_full_grid_load(self) -> None:
        """Plots the full grid load for the dataset."""
        self.__make_fig()
        plt.plot(self.df_orig['datetime'], self.df_orig["nat_demand"], label="nat demand", alpha=0.5)
        plt.xlabel("Date", fontsize=self.fontsize)
        plt.ylabel("National Demand [MWh]", fontsize=self.fontsize)
        plt.xlim(self.df_orig['datetime'].min(), self.df_orig['datetime'].max())
        plt.ylim(0, 1750)
        self.__set_x_and_y_ticks()
        plt.tight_layout()
        self.__save_fig('full_grid_load')
        plt.show()
        plt.close()

    def plot_grid_load_intra_day(self) -> None:
        """Plots weekly and daily intra-day averages of the grid load."""
        avg_daily = self.df_pre_covid.groupby('day_of_week')['nat_demand'].mean()
        avg_hourly = self.df_pre_covid.groupby('hour')['nat_demand'].mean()

        # Weekly average
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

        # Daily hourly average
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

    def plot_grid_load_full_dataset(self) -> None:
        """Plots the entire grid load dataset with low alpha for overview."""
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

    def plot_grid_load(self) -> None:
        """
        Decomposes pre-COVID grid load into trend, seasonality, and residuals,
        then plots each component.
        """
        window_size_4_weeks: int = 24 * 7 * 4  # 4 weeks ~ 1 month

        # Calculate trend, seasonality, residual
        self.df_pre_covid['monthly_trend'] = self.df_pre_covid['nat_demand'].rolling(window=window_size_4_weeks, center=True).mean()
        self.df_pre_covid['seasonality'] = (self.df_pre_covid['nat_demand'] - self.df_pre_covid['monthly_trend']).rolling(window=window_size_4_weeks, center=True).mean()
        self.df_pre_covid['residual'] = self.df_pre_covid['nat_demand'] - self.df_pre_covid['monthly_trend'] - self.df_pre_covid['seasonality']

        os.makedirs("images/seasonal_decomposition", exist_ok=True)

        # Plot each component
        self.__plot_seasonlity_figure(self.df_pre_covid['nat_demand'], "full_grid_load", "Original Data")
        self.__plot_seasonlity_figure(self.df_pre_covid['monthly_trend'], "trend", label="Trend (4-Weekly MA)", color='green', linewidth=3)
        self.__plot_seasonlity_figure(self.df_pre_covid['seasonality'], "seasonality", label='Seasonality', color='green', linewidth=3)
        self.__plot_seasonlity_figure(self.df_pre_covid['residual'], "residual", label='Residual', color='green', linewidth=1)

    def plot_holiday_IDs(self) -> None:
        """Plots grid load per holiday ID with error bars and average line."""
        avg_demand: float = self.df_pre_covid['nat_demand'].mean()

        grouped = self.df_pre_covid.groupby('Holiday_ID')['nat_demand']
        avg_holiday = grouped.mean()
        std_holiday = grouped.std()

        plt.figure(figsize=(10, 5))
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
