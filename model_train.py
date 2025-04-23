# model_train.py
import pandas as pd
import fastf1
from fastf1.api import Cache
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Enable FastF1 cache
Cache.enable_cache('./cache')

def load_race_data(year, gp):
    session = fastf1.get_session(year, gp, 'R')
    session.load()

    laps = session.laps
    drivers = session.drivers
    all_laps = []

    for drv in drivers:
        drv_laps = laps.pick_driver(drv)[['LapNumber', 'Compound', 'TyreLife', 'TrackStatus', 'IsAccurate']]
        drv_laps['Driver'] = drv
        all_laps.append(drv_laps)

    df = pd.concat(all_laps).reset_index(drop=True)
    df = df[df['IsAccurate']]
    df.drop(columns=['IsAccurate'], inplace=True)

    le = LabelEncoder()
    df['Compound'] = le.fit_transform(df['Compound'])

    return df

def train_model(df):
    if 'Compound' not in df.columns:
        raise ValueError("Compound column missing from data.")

    X = df.drop(columns=['Compound', 'Driver'])
    y = df['Compound']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

def get_weather(session):
    weather_data = session.weather_data.copy()
    keep_cols = ['Time', 'Rainfall', 'TrackTemp', 'AirTemp']
    return weather_data[keep_cols]
