# app.py
import streamlit as st
import pandas as pd
import fastf1
import altair as alt
from model_train import load_race_data, train_model, get_weather

# App config
st.set_page_config(layout="wide")
st.title("🏎️ F1 Pit Stop Strategy AI")

# User inputs
year = st.selectbox("Select Year", [2023, 2022, 2021])
gp = st.selectbox("Select Grand Prix", ['monaco', 'spa', 'silverstone', 'singapore', 'japan'])

# Button: Load & Predict
if st.button("Load & Predict"):
    with st.spinner("Loading session and data..."):
        session = fastf1.get_session(year, gp, 'R')
        session.load()
        df = load_race_data(year, gp)
        st.subheader("📊 Pit Stop Data")
        st.dataframe(df.head())

    with st.spinner("Getting weather data..."):
        weather_df = get_weather(session)

        # Convert durations to seconds for plotting
        if pd.api.types.is_timedelta64_dtype(weather_df['Time']):
            weather_df['Time'] = weather_df['Time'].dt.total_seconds()
        for col in weather_df.select_dtypes(include=['timedelta64[ns]']).columns:
            if col != 'Time':
                weather_df[col] = weather_df[col].dt.total_seconds()

        st.subheader("🌤️ Weather During Race (Track Temp)")
        chart = alt.Chart(weather_df).mark_line().encode(
            x=alt.X('Time', title='Time (seconds)'),
            y=alt.Y('TrackTemp', title='Track Temperature (°C)'),
            tooltip=['Time', 'TrackTemp', 'Rainfall', 'AirTemp']
        ).properties(
            width=800,
            height=400,
            title='Track Temperature Over Time'
        ).interactive()

        st.altair_chart(chart)

    with st.spinner("Training model..."):
        model = train_model(df)
    st.success("✅ Model trained!")

    # Example prediction (change according to your model's output structure)
    if hasattr(model, 'predict'):
        st.subheader("📈 Model Predictions")

        # Dummy feature for demo (replace with real ones)
        features = df.drop(columns=['Target'], errors='ignore')  # replace 'Target' as needed
        try:
            predictions = model.predict(features)
            df['Prediction'] = predictions
            st.dataframe(df[['Driver', 'Lap', 'Prediction']].head())
        except Exception as e:
            st.warning(f"Prediction error: {e}")

    # Tyre strategy recommendation
    avg_rain = weather_df['Rainfall'].mean()
    avg_temp = weather_df['TrackTemp'].mean()

    st.subheader("🛞 Recommended Tyre Type")
    if avg_rain > 0.1:
        st.info("🔵 Recommend: **Wet** or **Intermediate** tyres due to rain.")
    elif avg_temp < 25:
        st.info("🟡 Recommend: **Mediums** — cooler track temp, balance is key.")
    else:
        st.info("🔴 Recommend: **Softs** or **Mediums** for dry, hot race.")