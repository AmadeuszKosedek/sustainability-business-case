import logging
import datetime
from config import raw_data_path, eda_path, modeling_path, imputed_data_path, categorical_columns, rf_param_dist, hue_columns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


class Dataset:
    '''
    Class to load, preprocess, save and access the dataset.

    Parameters:
    - filepath: Dataset filepath.

    Methods:
    - load_data: Load the dataset from the filepath.
    - show_summary: Display a summary of the dataset.
    - impute_missing_values: Impute missing values in the dataset.
    - check_data_types: Check the data types in the dataset.
    - drop_categorical_columns: Drop categorical columns from the dataset.
    - save_data: Save the dataset to a new filepath.
    - get_data: Return the dataset.
    - get_numeric_columns: Return the numeric columns in the dataset.
    - get_categorical_columns: Return the categorical columns in the dataset.    
    '''
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.data = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.filepath)
            logging.info(f"Data loaded. Dataframe shape is: {self.data.shape}")
            logging.info("Initial data types:\n" + str(self.data.dtypes))
        except FileNotFoundError:
            logging.error(f"File not found: {self.filepath}")
            raise
        except pd.errors.EmptyDataError:
            logging.error("No data: The file is empty.")
            raise
        except Exception as e:
            logging.error(f"An error occurred while loading data: {e}")
            raise

    def show_summary(self):
        logging.info(self.data.info())

    def get_data_without_na(self):
        return self.data.dropna()
        
    def check_data_types(self):
        logging.info("Data types in DataFrame:")
        logging.info(self.data.dtypes)
        data_types = self.data.dtypes
        non_numeric_columns = data_types[~data_types.apply(lambda dtype: np.issubdtype(dtype, np.number))].index.tolist()
        if non_numeric_columns:
            logging.warning(f"Non-numeric columns found: {non_numeric_columns}")
            return non_numeric_columns
        else:
            logging.info("All columns are numeric.")
            return []

    def drop_categorical_columns(self):
        self.data = self.data.drop(columns=categorical_columns, axis=1)

    def save_data(self, filepath):
        self.data.to_csv(filepath, index=False)

    def get_data(self):
        return self.data

    def get_numeric_columns(self):
        return self.data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    def get_categorical_columns(self):
        return self.data.select_dtypes(include=['object']).columns.tolist()


class DataAnalysis:
    '''
    Class for performing data analysis and generating plots.

    Parameters:
    - data (pd.DataFrame): The input data for analysis.

    Methods:
    - plot_missing_values: Generates a heatmap plot to visualize missing values in the data.
    - plot_histograms: Generates histograms for each numeric variable in the data.
    - plot_scatter_matrix: Generates a scatter matrix plot for each numeric variable with an optional hue column.
    - plot_all_series: Generates time series plots for each numeric variable with an optional hue column.
    - check_stationarity: Checks the stationarity of each numeric variable in the data.
    - check_seasonality: Checks the seasonality of each numeric variable in the data.
    - plot_acf_pacf: Generates ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots for each numeric variable in the data.
    '''
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data 

    def plot_missing_values(self, path=eda_path):
        logging.info("Generating missing values plot.")
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.data.isnull(), cbar=False)
        plt.title('Missing Values in Data')
        plt.savefig(f'{eda_path}missing_values.png') 
        plt.close()
        logging.info(f"Missing values plot saved in {eda_path}")

    def plot_histograms(self, path=eda_path):
        logging.info("Generating histograms for each numeric variable.")
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[col], kde=True)
            plt.title(f'Histogram of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.savefig(f'{eda_path}histogram_{col}.png')
            plt.close()
        logging.info(f"Histograms saved in {eda_path}")

    def plot_scatter_matrix(self, hue=None, path=eda_path):
        for hue_column in hue:
            logging.info(f"Generating scatter matrix for each numeric variable with {hue_column}.")
            plt.figure(figsize=(12, 8))
            sns.pairplot(self.data, hue=hue_column)
            plt.title(f'Scatter Matrix of Numeric Variables with {hue_column}')
            plt.savefig(f'{eda_path}scatter_matrix_{hue_column}.png')
            plt.close()
        logging.info(f"Scatter matrix saved in {eda_path}")

    def plot_scatter_for_each_pair(self, hue=None):
        pass

        
    def plot_all_series(self, hue=None, date_column='reporting_date', path=eda_path):
        for hue_column in hue:
            logging.info(f"Generating time series plot for each numeric variable with {hue_column}.")
            for col in self.data.select_dtypes(include=['float64', 'int64']).columns:
                plt.figure(figsize=(12, 8))
                self.data[date_column] = pd.to_datetime(self.data[date_column]) 
                ax = sns.lineplot(x=date_column, y=col, data=self.data, hue=hue_column)
                plt.title(f'Time Series of {col} with {hue_column}')
    
                ax.xaxis.set_major_locator(mdates.AutoDateLocator()) 
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45) 
    
                plt.savefig(f'{eda_path}time_series_{hue_column}_{col}.png')
                plt.close()
            logging.info(f"Time series plots saved for {hue_column}.")

    def check_stationarity(self):
        logging.info("Checking stationarity for each numeric variable.")
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        results = {}
        for col in numeric_cols:
            result = adfuller(self.data[col].dropna())
            results[col] = {'ADF Statistic': result[0], 'p-value': result[1]}
            logging.info(f'Stationarity Test - {col}: ADF Statistic = {result[0]}, p-value = {result[1]}')
            if result[1] < 0.05:
                logging.info(f"{col} is stationary.")
            else:
                logging.info(f"{col} is not stationary.")
        return results
    
    def check_for_white_noise(self, lags=10, p_value_threshold=0.05, verbose=False):
        """
        Check if numeric variables in the dataset are white noise.
        
        Parameters:
            lags (int or list): Number of lags to use for the Ljung-Box test.
            p_value_threshold (float): Threshold for determining white noise based on p-value.
            verbose (bool): If True, log detailed information.

        Returns:
            dict: Results of the Ljung-Box test for each numeric column.
        """
        logging.info("Checking for white noise in each numeric variable.")
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        results = {}

        for col in numeric_cols:
            series = self.data[col].dropna()
            actual_lags = max(lags) if isinstance(lags, list) else lags

            if len(series) < actual_lags:
                logging.warning(f"Not enough data in {col} after dropping NA to perform Ljung-Box test.")
                results[col] = {'Error': f'Insufficient data after NA removal. Needed: {actual_lags}, Available: {len(series)}'}
                continue

            try:
                result = acorr_ljungbox(series, lags=[actual_lags], return_df=True)
                p_value = result['lb_pvalue'].iloc[0]
                results[col] = {'Ljung-Box p-value': p_value}

                conclusion = "likely white noise" if p_value > p_value_threshold else "not white noise"
                results[col]['Conclusion'] = conclusion
                if verbose:
                    logging.info(f"{col} is {conclusion} (p-value: {p_value}).")

            except Exception as e:
                logging.error(f"Error processing {col}: {str(e)}")
                results[col] = {'Error': str(e)}

        return results
    
    def check_seasonality(self):
        logging.info("Checking seasonality for each numeric variable.")
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='reporting_date', y=col, data=self.data)
            plt.title(f'Seasonality of {col}')
            plt.savefig(f'{eda_path}seasonality_{col}.png')
            plt.close()
        logging.info("Seasonality plots saved in the 'plots' folder.")
    
    def plot_acf_pacf(self):
        logging.info("Generating ACF and PACF plots.")
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            plot_acf(self.data[col].dropna(), lags=50, ax=ax[0])
            plot_pacf(self.data[col].dropna(), lags=50, ax=ax[1])
            ax[0].set_title(f'ACF of {col}')
            ax[1].set_title(f'PACF of {col}')
            plt.savefig(f'{eda_path}acf_pacf_{col}.png')
            plt.close()
        logging.info("ACF and PACF plots saved in the 'plots' folder.")

    def plot_qq_plot(self):
        logging.info("Generating QQ plots.")
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            sm.qqplot(self.data[col].dropna(), line ='45')
            plt.title(f'QQ Plot of {col}')
            plt.savefig(f'{eda_path}qq_plot_{col}.png')
            plt.close()
        logging.info("QQ plots saved in the 'plots' folder.")

    def plot_all_in_one_histogram(self):
        logging.info("Generating all-in-one histogram plot.")
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(self.data.select_dtypes(include=['float64', 'int64']).columns):
            plt.subplot(3, 2, i+1)
            sns.histplot(self.data[col], kde=True)
            plt.title(f'Histogram of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'{eda_path}all_in_one_histogram.png')
        plt.close()
        logging.info("All-in-one histogram plot saved in the 'plots' folder.")

    def plot_all_in_one_qq_plot(self):
        logging.info("Generating all-in-one QQ plot.")
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(self.data.select_dtypes(include=['float64', 'int64']).columns):
            plt.subplot(3, 2, i+1)
            sm.qqplot(self.data[col].dropna(), line ='45')
            plt.title(f'QQ Plot of {col}')
        plt.tight_layout()
        plt.savefig(f'{eda_path}all_in_one_qq_plot.png')
        plt.close()
        logging.info("All-in-one QQ plot saved in the 'plots' folder.")

    def add_carbon_footprint_energy_ratio(self):
        if 'carbon_footprint' in self.data.columns and 'energy_consumption' in self.data.columns:
            self.data['carbon_footprint_energy_ratio'] = self.data['carbon_footprint'] / self.data['energy_consumption']
            logging.info("Added 'carbon_footprint_energy_ratio' column to the dataset.")
        else:
            logging.error("Required columns 'carbon_footprint' and/or 'energy' not found in the dataset.")
            raise KeyError("Required columns 'carbon_footprint' and/or 'energy' not found in the dataset.")

    def plot_correlation_heatmap(self):
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.savefig(f'{eda_path}correlation_heatmap.png')
        plt.close()
        logging.info("Correlation heatmap saved in the 'plots' folder.")

    def plot_proportion_of_missing_values_over_time(self):
        logging.info("Generating proportion of missing values over time plot.")
        missing_values = self.data.isnull().sum(axis=1)
        missing_values = missing_values[missing_values > 0]
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=self.data['reporting_date'], y=missing_values)
        plt.title('Proportion of Missing Values Over Time')
        plt.savefig(f'{eda_path}missing_values_over_time.png')
        plt.close()
        logging.info("Proportion of missing values over time plot saved in the 'plots' folder.")

    def plot_correlation_heatmap(self):
        numeric_data = self.data.select_dtypes(include=['float64', 'int64'])
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.savefig(f'{eda_path}correlation_heatmap.png')
        plt.tight_layout()
        plt.close()
        logging.info("Correlation heatmap saved in the 'plots' folder.")

    def plot_proportion_of_missing_values_over_time(self):
        logging.info("Generating proportion of missing values over time plot.")
        missing_values = self.data.isnull().sum(axis=1)
        missing_values = missing_values[missing_values > 0]
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=self.data['reporting_date'], y=missing_values)
        plt.title('Proportion of Missing Values Over Time')
        plt.savefig(f'{eda_path}missing_values_over_time.png')
        plt.close()
        logging.info("Proportion of missing values over time plot saved in the 'plots' folder.")

    def check_seasonality_of_time_series(self):
        logging.info("Checking seasonality of time series data.")
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='reporting_date', y=col, data=self.data)
            plt.title(f'Seasonality of {col}')
            plt.savefig(f'{eda_path}seasonality_{col}.png')
            plt.close()
        logging.info("Seasonality plots saved in the 'plots' folder.")


class AutoArimaModel:
    def __init__(self, data):
        self.data = data
        self.model = None

    def train(self):
        logging.info("Training Auto ARIMA model...")
        self.model = auto_arima(self.data['carbon_footprint'], seasonal=True, m=12, stepwise=True, suppress_warnings=True)
        logging.info("Auto ARIMA model trained.")
        logging.info("Model summary:")
        logging.info(self.model.summary())

    def forecast(self, steps=12):
        if self.model is not None:
            logging.info(f"Forecasting {steps} steps ahead...")
            forecast = self.model.predict(n_periods=steps)
            logging.info("Forecasting complete.")
            logging.info(f"Forecast: {forecast}")

            return forecast
        else:
            logging.error("Model is not trained. Cannot forecast.")
            return None

    def plot_forecast(self, forecast, modeling_path):
        if forecast is not None:
            plt.figure(figsize=(12, 6))
            plt.plot(self.data['carbon_footprint'], label='Actual')
            plt.plot(self.data.index[-1] + np.arange(1, len(forecast) + 1), forecast, label='Forecast', linestyle='--')
            plt.legend()
            plt.title('Auto ARIMA Forecast')
            plt.xlabel('Time')
            plt.ylabel('Carbon Footprint')
            plt.savefig(f'{modeling_path}auto_arima_forecast.png')
            plt.close()
            logging.info("Forecast plot saved in the 'modeling' folder.")
        else:
            logging.error("No forecast data to plot.")

class RandomForestModel:
    def __init__(self, data):
        self.data = data
        self.model = None

    def train(self, rf_param_dist, n_iter=50, cv_folds=5):
        X = self.data[['production_volume', 'energy_consumption', 'water_usage', 'waste_generated', 'recycled_materials_ratio']]
        y = self.data['carbon_footprint']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        rf = RandomForestRegressor(random_state=42)
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=rf_param_dist, n_iter=n_iter, cv=cv_folds, verbose=2, random_state=42, n_jobs=-1)
        rf_random.fit(X_train, y_train)
        
        self.model = rf_random.best_estimator_
        y_pred_rf = self.model.predict(X_test)
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        r2_rf = r2_score(y_test, y_pred_rf)
        
        logging.info(f"Random Forest MSE: {mse_rf}")
        logging.info(f"Random Forest MAE: {mae_rf}")
        logging.info(f"Random Forest R2: {r2_rf}")
        logging.info("Best Random Forest parameters: {}".format(rf_random.best_params_))
    
    def current_timestamp(self):
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def perform_cross_validation(self, cv_folds=5):
        X = self.data[['production_volume', 'energy_consumption', 'water_usage', 'waste_generated', 'recycled_materials_ratio']]
        y = self.data['carbon_footprint']
        
        cv_scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring='r2')
        return cv_scores

    def plot_cv_results(self, cv_scores):
        timestamp = self.current_timestamp()
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=cv_scores)
        plt.title('Cross-Validation R2 Scores')
        plt.xlabel('R2 Score')
        plt.savefig(f'{modeling_path}rf_cv_results{timestamp}.png')
        plt.close()
        logging.info("Cross-validation results plot saved in the 'modeling' folder.")


    def plot_feature_importance(self):
        
        feature_importance = self.model.feature_importances_
        feature_names = ['production_volume', 'energy_consumption', 'water_usage', 'waste_generated', 'recycled_materials_ratio']
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importance, y=feature_names)
        plt.title('Feature Importance')
        plt.savefig(f'{modeling_path}rf_feature_importance.png')
        plt.close()
        logging.info("Feature importance plot saved in the 'modeling' folder.")

    def plot_residuals(self):
        X = self.data[['production_volume', 'energy_consumption', 'water_usage', 'waste_generated', 'recycled_materials_ratio']]
        y = self.data['carbon_footprint']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        if self.model is not None:
            y_pred = self.model.predict(X_test)
            residuals = y_test - y_pred

            plt.figure(figsize=(10, 6))
            sns.residplot(x=y_pred, y=residuals, lowess=True, scatter_kws={'alpha': 0.5})
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            plt.savefig(f'{modeling_path}rf_residual_plot.png')
            plt.close()
            logging.info("Residual plot saved in the 'modeling' folder.")
        else:
            logging.error("Model is not trained. Cannot plot residuals.")

    def plot_predictions(self):
        X = self.data[['production_volume', 'energy_consumption', 'water_usage', 'waste_generated', 'recycled_materials_ratio']]
        y = self.data['carbon_footprint']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        if self.model is not None:
            y_pred = self.model.predict(X_test)

            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
            plt.xlabel('Measured')
            plt.ylabel('Predicted')
            plt.title('Comparison of Predictions and Actual Values')
            plt.savefig(f'{modeling_path}rf_prediction_comparison.png')
            plt.close()
            logging.info("Prediction comparison plot saved in the 'modeling' folder.")
        else:
            logging.error("Model is not trained. Cannot plot predictions.")


class KMeansClustering:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.model = None
        self.n_clusters = None

    def train(self, n_clusters=3, random_state=42):
        X = self.data[['production_volume', 'energy_consumption', 'water_usage', 'waste_generated', 'recycled_materials_ratio']]
        X = StandardScaler().fit_transform(X) 
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.model.fit(X)
        self.data['cluster'] = self.model.labels_
        self.n_clusters = n_clusters
        logging.info(f"KMeans trained with {n_clusters} clusters.")

    def fit_predict(self, data):
        X = data[['production_volume', 'energy_consumption', 'water_usage', 'waste_generated', 'recycled_materials_ratio']]
        X = StandardScaler().fit_transform(X)
        return self.model.predict(X)

    def plot_clusters(self):
        if self.model is not None:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='production_volume', y='energy_consumption', data=self.data, hue='cluster', palette='viridis', style='cluster')
            plt.title('KMeans Clustering')
            plt.savefig(f'{modeling_path}best_kmeans_clusters.png')
            plt.close()
            logging.info(f"KMeans clustering plot saved in the {modeling_path}")
        else:
            logging.error("Model is not trained. Cannot plot clusters.")

    def calculate_cluster_metrics(self):
        if self.model is not None:
            cluster_metrics = {}
            for cluster in range(self.n_clusters):
                cluster_data = self.data[self.data['cluster'] == cluster]
                cluster_metrics[f'Cluster {cluster}'] = {
                    'Mean Production Volume': cluster_data['production_volume'].mean(),
                    'Mean Energy Consumption': cluster_data['energy_consumption'].mean(),
                    'Mean Water Usage': cluster_data['water_usage'].mean(),
                    'Mean Waste Generated': cluster_data['waste_generated'].mean(),
                    'Mean Recycled Materials Ratio': cluster_data['recycled_materials_ratio'].mean()
                }
            return cluster_metrics  
        else:
            logging.error("Model is not trained. Cannot calculate cluster metrics.")
            return None

                
    def auto_tune_parameters(self, n_clusters_range):
        X = self.data[['production_volume', 'energy_consumption', 'water_usage', 'waste_generated', 'recycled_materials_ratio']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        best_silhouette_score = -1
        best_davies_bouldin_score = float('inf')
        best_n_clusters = None
        tuning_results = []

        for n_clusters in n_clusters_range:
            model = KMeans(n_clusters=n_clusters)
            labels = model.fit_predict(X_scaled)
            
            silhouette = silhouette_score(X_scaled, labels)
            davies_bouldin = davies_bouldin_score(X_scaled, labels)
            
            tuning_results.append((n_clusters, silhouette, davies_bouldin))
            logging.info(f"n_clusters: {n_clusters}, Silhouette Score: {silhouette}, Davies-Bouldin Index: {davies_bouldin}")

            if silhouette > best_silhouette_score and davies_bouldin < best_davies_bouldin_score:
                best_silhouette_score = silhouette
                best_davies_bouldin_score = davies_bouldin
                best_n_clusters = n_clusters

        if best_n_clusters:
            self.train(best_n_clusters)
            return best_n_clusters, tuning_results
        else:
            logging.warning("No valid clustering found within the given parameter ranges.")
            return None, tuning_results


    def plot_tuning_results(self, tuning_results):
        n_clusters, silhouettes, davies_bouldins = zip(*tuning_results)
        
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Silhouette Score', color=color)
        ax1.plot(n_clusters, silhouettes, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Davies-Bouldin Index', color=color)
        ax2.plot(n_clusters, davies_bouldins, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('KMeans Tuning Results')
        fig.tight_layout()
        plt.savefig(f'{modeling_path}kmeans_tuning_results.png')
        plt.close()
        logging.info("KMeans tuning results plot saved.")


def main():

    # ---------- Load the data
    dataset = Dataset(raw_data_path)
    dataset.load_data()

    # --------- Perform EDA on the raw data
    analysis = DataAnalysis(dataset.get_data())
    analysis.plot_missing_values()
    analysis.plot_proportion_of_missing_values_over_time()
    analysis.add_carbon_footprint_energy_ratio()
    analysis.plot_histograms()
    #analysis.plot_all_in_one_histogram()
    analysis.plot_qq_plot()
    #analysis.plot_all_in_one_qq_plot()
    analysis.plot_scatter_matrix(hue=hue_columns)
    analysis.plot_all_series(hue=hue_columns)
    analysis.plot_acf_pacf()
    analysis.check_stationarity()
    analysis.check_seasonality()
    analysis.check_for_white_noise(lags=10, p_value_threshold=0.05, verbose=True)
    analysis.plot_correlation_heatmap()
    analysis.add_carbon_footprint_energy_ratio()

    dataset_filtered = dataset.get_data()
    dataset_filtered = dataset_filtered[dataset_filtered['carbon_footprint_energy_ratio'] < 0.3]

    # ---------- Below similar procedure but on filtered data
    #analysis = DataAnalysis(dataset_filtered)
    #analysis.plot_missing_values()
    #analysis.plot_proportion_of_missing_values_over_time()
    #analysis.plot_histograms()
    #analysis.plot_qq_plot()
    #analysis.plot_scatter_matrix(hue=hue_columns)
    #analysis.plot_all_series(hue=hue_columns)
    #analysis.plot_acf_pacf()
    #analysis.check_stationarity()
    #analysis.check_seasonality()
    #analysis.check_for_white_noise(lags=10, p_value_threshold=0.05, verbose=True)

    dataset_without_na = dataset.get_data_without_na()
 



    #analysis.plot_correlation_heatmap()

    # ---------- Attempt to run KMeans clustering
    kmeans_model = KMeansClustering(dataset_without_na)
    best_n_clusters, tuning_results = kmeans_model.auto_tune_parameters(n_clusters_range=range(2, 10))
    if best_n_clusters:
        kmeans_model.train(n_clusters=best_n_clusters)
        kmeans_model.plot_clusters()

    kmeans_model.plot_tuning_results(tuning_results)
    kmeans_model.calculate_cluster_metrics( )
    cluster_labels = kmeans_model.fit_predict(dataset_without_na)
    dataset_without_na['cluster'] = cluster_labels

    # ---------- Unnecessary model - fit RF for each best cluster

    #for cluster in range(2):
    #    cluster_data = dataset_filtered[dataset_filtered['cluster'] == cluster].drop('cluster', axis=1)
    #    
    #    rf_model = RandomForestModel(cluster_data)
    #    rf_model.train(rf_param_dist)
    #    rf_model.plot_feature_importance()
    #    rf_model.plot_residuals()
    #    rf_model.plot_predictions()   
    #    rf_model.cv_scores = rf_model.perform_cross_validation()
    #    rf_model.plot_cv_results(rf_model.cv_scores)
    



    logging.info("All tasks completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M',
                    handlers=[logging.StreamHandler()])

    main()