"""Clean gym workout data and save a validated CSV.

Usage:
    python clean_gym_data.py
    python clean_gym_data.py --input gym_members_exercise_tracking.csv --output gym_cleaned.csv

Missing values are intentionally left for a training-only preprocessing pipeline
so medians and category imputations cannot leak test-set information.
"""
import argparse
from pathlib import Path
import pandas as pd

TARGET = 'Max_BPM'
CATEGORICAL = ['Gender', 'Workout_Type']
NUMERIC = [
    'Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM',
    'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned',
    'Fat_Percentage', 'Water_Intake (liters)',
    'Workout_Frequency (days/week)', 'Experience_Level', 'BMI',
]
REQUIRED = NUMERIC + CATEGORICAL


def clean_data(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    missing_columns = sorted(set(REQUIRED) - set(df.columns))
    if missing_columns:
        raise ValueError(f'Missing required columns: {missing_columns}')

    print(f'Original shape: {df.shape}')
    df = df.drop_duplicates().copy()
    for column in NUMERIC:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    for column in CATEGORICAL:
        df[column] = df[column].astype('string').str.strip()
        df[column] = df[column].replace('', pd.NA)

    # Physically impossible or clearly invalid values become missing.
    ranges = {
        'Age': (1, 120),
        'Weight (kg)': (1, 500),
        'Height (m)': (0.5, 2.7),
        'Max_BPM': (30, 250),
        'Avg_BPM': (30, 250),
        'Resting_BPM': (20, 200),
        'Session_Duration (hours)': (0, 24),
        'Calories_Burned': (0, 10000),
        'Fat_Percentage': (0, 100),
        'Water_Intake (liters)': (0, 30),
        'Workout_Frequency (days/week)': (0, 7),
        'Experience_Level': (1, 3),
        'BMI': (5, 100),
    }
    for column, (lower, upper) in ranges.items():
        df.loc[~df[column].between(lower, upper), column] = float('nan')

    # A session average cannot exceed its maximum; preserve the row but
    # mark the inconsistent input as missing, to be imputed after splitting.
    inconsistent = df['Avg_BPM'] > df['Max_BPM']
    df.loc[inconsistent, 'Avg_BPM'] = float('nan')

    # Never impute the prediction target: it must be observed.
    df = df.dropna(subset=[TARGET]).copy()
    if df.empty:
        raise ValueError('No rows with valid Max_BPM remain.')

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f'Cleaned shape: {df.shape}')
    print('Missing values by column (handled during model training):')
    print(df.isna().sum().to_string())
    print(f'Saved to: {output}')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='gym_members_exercise_tracking.csv')
    parser.add_argument('--output', default='gym_cleaned.csv')
    args = parser.parse_args()
    clean_data(args.input, args.output)
