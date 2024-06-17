raw_data_path = "/home/amadeusz/projects/reckitt/data/sustainability_data.csv"
raw_data_without_na_path = (
    "/home/amadeusz/projects/reckitt/data/sustainability_data_without_na.csv"
)
imputed_data_path = (
    "/home/amadeusz/projects/reckitt/data/sustainability_data_imputed.csv"
)

eda_path = "/home/amadeusz/projects/reckitt/reporting/EDA/" # !!! this is the path where the EDA plots will be saved
modeling_path = "/home/amadeusz/projects/reckitt/reporting/modeling/" # !!! this is the path where the modeling plots will be saved


categorical_columns = ["factory_location", "product_category"]

numerical_columns = [
    "production_volume",
    "energy_consumption",
    "water_usage",
    "waste_generated",
    "recycled_materials_ratio",
]

hue_columns = ["product_category", "factory_location"]


rf_param_dist = {
            'n_estimators': [100, 200, 300, 400, 500, 1000],
            'max_features': [x for x in range(1, 6)],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }