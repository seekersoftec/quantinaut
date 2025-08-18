# From https://github.com/mbsuraj/stationarityToolkit/
import logging
import pickle
import numpy as np
import pandas as pd
from arch.unitroot import PhillipsPerron
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox


class Stationarity:
    def __init__(self, alpha, timeseries=None):
        """
        Constructor for the Stationarity class.

        Parameters:
            alpha (float): The significance level for hypothesis testing. A value between 0 and 1.
            timeseries (pandas.DataFrame): The input time series data in DataFrame format.
        """
        self._initiate_logger()
        self.alpha = alpha
        self.timeseries = timeseries
        self._recurse_cnt = 0
        self._differencing = None
        self._trend_initial_value = None
        self._seasonality_initial_value = None
        self._index = self._get_index()
        self._var_nonstationarity_removed = False
        self.df = None

    def _initiate_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create a formatter with a custom format
        formatter = logging.Formatter("[%(levelname)s] - %(message)s")

        # Create a console handler and set the formatter
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Add the console handler to the logger
        self.logger.addHandler(console_handler)

    def _get_index(self):
        if self.timeseries is not None:
            return self.timeseries.index
        else:
            return None

    def perform_pp_test(self, ts):
        """
        Perform the Phillips-Perron test on the given time series to test for variance stationarity.

        Parameters:
            ts (pandas.Series): The input time series data as a pandas Series.

        Returns:
            float: The p-value obtained from the Phillips-Perron test.
        """
        # Check if the logger has any handlers already
        if self.logger.hasHandlers():
            # If the logger already has handlers, remove them
            self.logger.handlers.clear()
        # Perform Dickey-Fuller test:
        self.logger.info("PHILLIPS-PERRON TEST FOR VARIANCE STATIONARITY")
        pp_test = PhillipsPerron(ts)
        p_value = pp_test.pvalue
        self.logger.debug(p_value)
        if p_value <= self.alpha:
            self.logger.info("The time series is likely variance stationary.")
        else:
            self.logger.info("The time series is likely not variance stationary.")
        return p_value

    def inv_boxcox(self, y_box, index, lambda_, constant):
        """
        Apply the inverse Box-Cox transformation to the given Box-Cox transformed data.

        Parameters:
            y_box (numpy.ndarray): The Box-Cox transformed data.
            lambda_ (float): The lambda value used in the Box-Cox transformation.
            index (pandas.Index): The index of the original time series.

        Returns:
            pandas.DataFrame: The time series in its original scale.
        """
        # Check if the logger has any handlers already
        if self.logger.hasHandlers():
            # If the logger already has handlers, remove them
            self.logger.handlers.clear()
        # Convert to numpy array
        y_box = np.array(y_box)
        ts = pd.DataFrame(np.power((y_box * lambda_) + 1, 1 / lambda_) - constant, index=index)
        return ts

    def remove_var_nonstationarity(self, ts=None):
        """
        Attempt to remove variance non-stationarity from the given time series.

        Parameters:
            ts (pandas.DataFrame, optional): The input time series data in DataFrame format.
                If None, the class's 'timeseries' attribute is used.

        Returns:
            pandas.DataFrame: DataFrame containing the original and transformed time series along with
                the transformation name, parameters, and inverse transformation function.
        """
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        ts = ts if ts is not None else self.timeseries
        self._index = self._get_index()

        self.logger.info("Test Variance Stationarity: ")
        if self.perform_pp_test(ts) > self.alpha:
            transformations = {
                "Original": (ts, None),
                "Log-Transformed": (np.log(ts), np.exp),
                "Square Root Transformed": (np.sqrt(ts), np.square),
            }

            # Box-Cox requires positive data, so we add a constant to the data to make it positive
            constant = (
                1  # Choose a positive constant (can be adjusted based on the data)
            )
            bc_ts = ts.values.flatten() + constant
            if np.all(bc_ts > 0):
                boxcox_transformed_data, lam = boxcox(bc_ts)
                transformations["Box-Cox Transformed"] = (
                    boxcox_transformed_data,
                    lambda x, idx: self.inv_boxcox(x, idx, lam, constant),
                )

            # Test variance stationarity for each transformed series
            best_transformation = None
            best_p_value = np.inf

            for name, (transformed_data, _) in transformations.items():
                self.logger.info(f"\n\ntesting {name}")
                p_value = self.perform_pp_test(transformed_data)
                if p_value <= best_p_value:
                    best_p_value = p_value
                    best_transformation = (name, transformed_data)

            # Extract the best transformation information
            best_transformation_name, best_transformed_data = best_transformation
            var_transformed = best_transformed_data if best_transformation_name == "Box-Cox Transformed" \
                else best_transformed_data.to_numpy().flatten()

            self.logger.info(f"\n\nBest Transformation: {best_transformation_name}")
            self.logger.info(f"P-Value: {best_p_value}")

            parameters = (
                {}
                if best_transformation_name != "Box-Cox Transformed"
                else str({"constant": constant, "lam": lam})
            )

            # Serialize the inverse function
            inv_function_serialized = transformations[best_transformation_name][1]

            # Plot the original and best transformed series
            self.df = pd.DataFrame(
                {
                    "original": ts.to_numpy().flatten(),
                    "var_transformed": var_transformed,
                    "var_transformation_name": best_transformation_name,
                    "var_transformation_par": parameters,
                    "var_inverse_function": inv_function_serialized,
                },
                index=ts.index,
            )
            plt.figure(figsize=(10, 6))
            plt.plot(self.df.index, self.df.original, label="Original Plot")
            plt.plot(self.df.index, self.df.var_transformed,
                     label=f"{best_transformation_name} Variance Stationary Plot")
            plt.title("Variance Stationarity Plot")
            # self.df.plot(
            #     title="Variance Stationarity with Best Transformation", figsize=(10, 6)
            # )
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Values")
            plt.show()
        else:
            self.df = pd.DataFrame(
                {
                    "original": ts.to_numpy().flatten(),
                    "var_transformed": ts.to_numpy().flatten(),
                    "var_transformation_name": None,
                    "var_transformation_par": None,
                    "var_inverse_function": None,
                },
                index=ts.index,
            )
        self._var_nonstationarity_removed = True
        return self.df

    def load_inverse_function(self, row):
        """
        Load the inverse transformation function from the serialized form in the DataFrame.

        Parameters:
            row (pandas.Series): A row from the DataFrame containing the serialized inverse function.

        Returns:
            Callable: The deserialized inverse transformation function.
        """
        inv_function_serialized = row["var_inverse_function"]
        if pd.notna(inv_function_serialized):
            return pickle.loads(inv_function_serialized)
        return None

    def adf_test(self, timeseries):
        """
        Perform the Augmented Dickey-Fuller (ADF) test on the given time series to test for trend stationarity.

        Parameters:
            timeseries (pandas.Series): The input time series data as a pandas Series.

        Returns:
            pandas.Series: Series containing the test results.
        """
        # Check if the logger has any handlers already
        if self.logger.hasHandlers():
            # If the logger already has handlers, remove them
            self.logger.handlers.clear()
        self.logger.info("DICKEY-FULLER TEST FOR TREND STATIONARITY")
        dftest = adfuller(timeseries, autolag="AIC")
        dfoutput = pd.Series(
            dftest[0:4],
            index=[
                "Test Statistic",
                "p-value",
                "#Lags Used",
                "Number of Observations Used",
            ],
        )
        for key, value in dftest[4].items():
            dfoutput["Critical Value (%s)" % key] = value

        self.logger.debug(dfoutput)
        if dfoutput["p-value"] <= self.alpha:
            self.logger.info("Reject Null Hypothesis: Series is stationary")
        else:
            self.logger.info(
                "Fail to reject Null Hypothesis: Series is non-stationary."
            )
        return dfoutput

    def kpss_test(self, timeseries):
        """
        Perform the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test on the given time series to test for trend stationarity.

        Parameters:
            timeseries (pandas.Series): The input time series data as a pandas Series.

        Returns:
            pandas.Series: Series containing the test results.
        """
        # Check if the logger has any handlers already
        if self.logger.hasHandlers():
            # If the logger already has handlers, remove them
            self.logger.handlers.clear()
        self.logger.info("KPSS TEST FOR TREND STATIONARITY")
        kpsstest = kpss(timeseries, regression="c")
        kpss_output = pd.Series(
            kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
        )
        for key, value in kpsstest[3].items():
            kpss_output["Critical Value (%s)" % key] = value
        self.logger.debug(kpss_output)
        if kpss_output["p-value"] <= self.alpha:
            self.logger.info("Reject Null Hypothesis: Series is non-stationary")
        else:
            self.logger.info(
                "Fail to reject Null Hypothesis: Series is trend-stationary."
            )
        return kpss_output

    def remove_trend_nonstationarity(self, ts=None):
        """
        Attempt to remove trend-based non-stationarity from the given time series.

        Parameters:
            ts (pandas.DataFrame, optional): The input time series data in DataFrame format.
                If None, the class's 'timeseries' attribute is used.

        Returns:
            tuple: A tuple containing the differenced time series, the type of differencing applied,
                and the initial value of the time series.
        """

        def seasonal_inv(diff_ts, seasonal_initial_values):
            val_len = len(diff_ts)
            for i in range(val_len):
                # print(i)
                # print(diff_ts.iloc[i, 0])
                if i < 52:
                    diff_ts.iloc[i] = seasonal_initial_values.iloc[i]
                else:
                    # Calculate the sum of every 52nd value
                    sum_52nd = diff_ts.iloc[i - 52] + diff_ts.iloc[i]
                    diff_ts.iloc[i] = sum_52nd
            return diff_ts

        def season_trend_inv(diff_ts2, seasonal_initial_values, trend_initial_value):
            inv_diff_ts2 = diff_ts2.cumsum() + trend_initial_value
            inv_diff_ts2.iloc[-1] = trend_initial_value
            inv_inv_diff_ts2 = seasonal_inv(inv_diff_ts2, seasonal_initial_values)
            return inv_inv_diff_ts2

        if self._recurse_cnt == 0:
            self.timeseries = ts if ts is not None else self.timeseries
            self._trend_initial_value = self.timeseries.values[0][0]
            self._index = self._get_index()

        if self._recurse_cnt == 0:
            # Check if the logger has any handlers already
            if self.logger.hasHandlers():
                # If the logger already has handlers, remove them
                self.logger.handlers.clear()
            self.logger.info("REMOVE TREND NON-STATIONARITY")
            self.logger.info("-----------------------LOG-------------------------")
            self.logger.info("INITIAL STATISTICAL TESTS")
        # Perform ADF and KPSS tests
        try:
            dfoutput = self.adf_test(ts.dropna())
            kpss_output = self.kpss_test(ts.dropna())
        except ValueError as e:
            self.logger.error(f"Error occurred during ADF or KPSS test: {e}")
            return None
        self._recurse_cnt += 1
        self.logger.info("----------------------------------------------------")
        self.logger.info(f"Recurse Count: {self._recurse_cnt}")
        self.logger.info("----------------------------------------------------")

        if (
            (dfoutput["p-value"] >= self.alpha)
            or (kpss_output["p-value"] <= self.alpha)
        ) and (self._recurse_cnt == 1):
            self.logger.info(
                "Both tests conclude that the series is not stationary -> Removing trend**"
            )
            self._differencing = "trend"
            self._trend_initial_value = ts.values[0][0]
            ts_dif = ts - ts.shift(1)
            if self.remove_trend_nonstationarity(ts_dif) is None:
                self.logger.info("Trend Removal didn't work. Removing Seasonality")
                self._differencing = "seasonality"
                self._seasonality_initial_value = ts.iloc[0:52].copy()
                ts_seasonal_diff = ts - ts.shift(52)
                if self.remove_trend_nonstationarity(ts_seasonal_diff) is None:
                    self._differencing = "seasonal_trend"
                    if len(ts_seasonal_diff) > 52:
                        self._trend_initial_value = ts_seasonal_diff.values[52]
                    else:
                        self.logger.error("ts_seasonal_diff doesn't have enough elements.")
                        return None
                    self.logger.info(
                        "Seasonality Removal didn't work. Removing Trend on top of Seasonal Differencing"
                    )
                    ts_seasonal_trend_diff = ts_seasonal_diff - ts_seasonal_diff.shift(
                        1
                    )
                    ts_seasonal_trend_diff.index = self._index
                    result = self.remove_trend_nonstationarity(ts_seasonal_trend_diff)
                    self._recurse_cnt = 0
                    return result
                else:
                    self.remove_trend_nonstationarity(ts_seasonal_diff)
                    self._recurse_cnt = 0
            else:
                result = self.remove_trend_nonstationarity(ts_dif)
                self._recurse_cnt = 0
                return result
            ts_dif.index = self._index
        elif (
            (dfoutput["p-value"] >= self.alpha)
            or (kpss_output["p-value"] <= self.alpha)
        ) and (self._recurse_cnt == 2):
            self._recurse_cnt = 1
            return None
        elif (dfoutput["p-value"] <= self.alpha) and (
            kpss_output["p-value"] >= self.alpha
        ):
            self.logger.info(
                f"**After {self._recurse_cnt} iterations - Both tests now conclude that the series is stationary**"
            )
            df = pd.DataFrame(
                {
                    "original": self.timeseries.to_numpy().flatten(),
                    "trend_transformed": ts.to_numpy().flatten(),
                    "trend_transformation_name": self._differencing,
                    "trend_initial_value": self._trend_initial_value,
                    "seasonal_initial_values": [self._seasonality_initial_value.values] if self._seasonality_initial_value is not None else None,
                    "trend_inverse_function": None,  # Placeholder for storing inverse function
                },
                index=ts.index,
            )
            if self.df is None:
                self.df = df
            elif (self.df is not None) and ("trend_transformed" not in self.df.columns):
                df.drop(columns=["original"], inplace=True)
                self.df = pd.merge(self.df, df, left_index=True, right_index=True)
            else:
                pass
            # Generate and save the inverse function
            inv_function_serialized = None
            if self._differencing == "trend":
                inv_function_serialized = (
                    lambda ts_diff: ts_diff.fillna(0).cumsum() + self._trend_initial_value
                )
            elif self._differencing == "seasonality":
                inv_function_serialized = lambda ts_diff: seasonal_inv(
                    ts_diff, self._seasonality_initial_value
                )
            elif self._differencing == "seasonal_trend":
                inv_function_serialized = lambda ts_diff: season_trend_inv(
                    ts_diff, self._seasonality_initial_value, self._trend_initial_value
                )

            self.df["trend_inverse_function"] = inv_function_serialized
            return self.df

    def _inverse_difference_fn(self, ts_diff, initial_value):
        """
        Inverse function for seasonal differencing ts - ts.shift(52).

        Parameters:
            ts_diff (pandas.Series): Differenced time series to be inverted.
            initial_value (float): The initial value of the original time series before differencing.

        Returns:
            pandas.Series: The inverted time series.
        """
        ts_inv = ts_diff.cumsum() + initial_value
        return ts_inv

    # # Serialize the inverse function using pickle
    # inv_function_serialized = pickle.dumps(lambda ts_diff: inverse_seasonal_difference(ts_diff, initial_value_example))
    def _plot_trend_stationary_series(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.df.index, self.df.original, label="Original")
        plt.plot(self.df.index, self.df.trend_transformed, label=f"{self._differencing}-Transformed")
        plt.plot(title="Trend Stationarity", figsize=(10, 6))
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Values")
        plt.show()

    def remove_nonstationarity(self, ts):
        """
        Perform both variance and trend-based non-stationarity removal from the input time series.

        Returns:
            pandas.DataFrame: DataFrame containing the original time series, the best variance-transformed time series,
                the differenced time series, the type of differencing applied, and the initial value of the time series.
        """
        self._recurse_cnt = 0
        self.timeseries = ts if ts is not None else self.timeseries
        self._index = self._get_index()
        df = self.remove_var_nonstationarity(ts)
        df2 = self.remove_trend_nonstationarity(df[["var_transformed"]])
        self._plot_trend_stationary_series()
        return df2
    