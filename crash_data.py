# This file manages the csv and turns it into data/features for the model
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load dataset
def load_dataset(filename):
    data = pd.read_csv(filename)
    return data

# Converts hour into a time block.
def get_time_block(hour):
    if 0 <= hour <= 5:
        return "late_night"
    elif 6 <= hour <= 9:
        return "morning_rush"
    elif 10 <= hour <= 15:
        return "midday"
    elif 16 <= hour <= 19:
        return "evening_rush"
    else:
        return "night"

# Converts crash count into risk target.
def count_to_risk_target(count):
    if count == 0:
        return 0  # NONE
    elif count == 1:
        return 1  # LOW
    elif count <= 3:
        return 2  # MEDIUM
    else:
        return 3  # HIGH

# 2. Aggregate hourly rows into time-block rows
def create_time_block_dataset(data):
    data["time_block"] = data["hour"].apply(get_time_block)

    time_block_codes = {
        "late_night": 0,
        "morning_rush": 1,
        "midday": 2,
        "evening_rush": 3,
        "night": 4,
    }

    data["time_block_code"] = data["time_block"].map(time_block_codes)
    aggregation_rules = {
        "date_index": "first",
        "year": "first",
        "month": "first",
        "month_sin": "first",
        "month_cos": "first",
        "day_of_month": "first",
        "day_of_week_num": "first",
        "day_of_week_sin": "first",
        "day_of_week_cos": "first",
        "is_weekend": "first",
        "is_major_us_holiday": "first",
        "is_holiday_window": "first",
        "direction_code": "first",
        "time_block_code": "first",
        "median_daily_traffic": "first",
        "median_adjusted_daily_traffic": "first",
        "median_speed_limit": "first",
        "median_number_of_lanes": "first",
        "median_latitude_for_direction_year": "first",
        "median_longitude_for_direction_year": "first",
        "lag_1h_crashes": "mean",
        "lag_2h_crashes": "mean",
        "lag_3h_crashes": "mean",
        "lag_24h_crashes": "mean",
        "lag_168h_crashes": "mean",
        "rolling_6h_avg_crashes": "mean",
        "rolling_24h_avg_crashes": "mean",
        "rolling_168h_avg_crashes": "mean",
        "rolling_720h_avg_crashes": "mean",
        "same_hour_last_7d_avg_crashes": "mean",
        "previous_24h_total_crashes": "mean",
        "previous_168h_total_crashes": "mean",
        # This is summed because the target is total crashes in the block.
        "crash_count": "sum",
    }
    block_data = data.groupby(["date", "direction", "time_block"], as_index=False).agg(aggregation_rules)
    block_data["risk_target"] = block_data["crash_count"].apply(count_to_risk_target)

    return block_data

# 3. Prepare X and y
def prepare_features_and_target(block_data):
    feature_columns = [
        "date_index",
        "year",
        "month",
        "month_sin",
        "month_cos",
        "day_of_month",
        "day_of_week_num",
        "day_of_week_sin",
        "day_of_week_cos",
        "is_weekend",
        "is_major_us_holiday",
        "is_holiday_window",
        "direction_code",
        "time_block_code",
        "median_daily_traffic",
        "median_adjusted_daily_traffic",
        "median_speed_limit",
        "median_number_of_lanes",
        "median_latitude_for_direction_year",
        "median_longitude_for_direction_year",
        "lag_1h_crashes",
        "lag_2h_crashes",
        "lag_3h_crashes",
        "lag_24h_crashes",
        "lag_168h_crashes",
        "rolling_6h_avg_crashes",
        "rolling_24h_avg_crashes",
        "rolling_168h_avg_crashes",
        "rolling_720h_avg_crashes",
        "same_hour_last_7d_avg_crashes",
        "previous_24h_total_crashes",
        "previous_168h_total_crashes",
    ]

    X = block_data[feature_columns].values
    y = block_data["risk_target"].values

    return X, y

# 4. Split dataset into training and test sets
def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
