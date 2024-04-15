import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def calculate_vehicle_speed(row):
    base_speed = 60  
    traffic_factor = -0.5 * (row['Verkehrsdichte'] / 100)  
    time_factor = -5 if row['Tageszeit'] == 1 else 0  
    weather_impact = (row['Wetterbedingungen'] - 10) * -1  
    weather_factor = weather_impact * 1.5  
    
    return max(10, base_speed + traffic_factor + time_factor + weather_factor)  

def calculate_braking_distance(row):
    base_distance = 20
    speed_factor = row['Fahrzeuggeschwindigkeit'] / 10
    road_condition_impact = (row['Straßenzustand'] - 10) * -1
    road_condition_factor = 1 + road_condition_impact * 0.05  
    weather_impact = (row['Wetterbedingungen'] - 10) * -1
    weather_factor = 1 + weather_impact * 0.05  
    
    return base_distance + (speed_factor * road_condition_factor * weather_factor) + np.random.normal(0, 2) 

def plot_data(X, y, model=None, lowess_results=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='skyblue', edgecolors='w', linewidth=0.5, s=50, label='Data points')
    plt.xlabel('X Werte', fontsize=14)
    plt.ylabel('Y Werte', fontsize=14)
    plt.title('Regressionsanalyse', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if model is not None:
        line_X = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
        line_y = model.predict(line_X)
        plt.plot(line_X, line_y, color='red', linewidth=2, label='Lineare Regression')

    if lowess_results is not None:
        plt.plot(lowess_results[:, 0], lowess_results[:, 1], color='green', linewidth=2, label='"Ideale" Regression')

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

def main():
    st.title('Beispiel: Statistical Modeling im Autonomen Fahrzeug')
    
    st.image('auto.jpg', caption='Autonomes Fahrzeug im Einsatz')
    
    np.random.seed(42)  

    data = pd.DataFrame({
        'Verkehrsdichte': np.random.uniform(10, 100, 100),
        'Tageszeit': np.random.choice(['Tag', 'Nacht'], 100),
        'Wetterbedingungen': np.random.uniform(1, 10, 100),  # 1 schlecht, 10 ausgezeichnet
        'Straßenzustand': np.random.uniform(1, 10, 100),  # 1 schlecht, 10 ausgezeichnet
        'Fahrzeuglast': np.random.normal(300, 50, 100),
        'Reifenzustand': np.random.uniform(1, 10, 100)  # 1 schlecht, 10 ausgezeichnet
    }) 

    data['Fahrzeuggeschwindigkeit'] = data.apply(calculate_vehicle_speed, axis=1)
    data['Bremsweg'] = data.apply(calculate_braking_distance, axis=1)
    
    available_columns = [col for col in data.columns if col != 'Tageszeit']
 
    selected_x = st.selectbox('Wähle unabhängige Variable (X):', options=data.columns)
    selected_y = st.selectbox('Wähle abhängige Variable (Y):', options=available_columns)
  
    add_linear_regression = st.checkbox('Lineare Regressionslinie hinzufügen', value=False)
    add_ideal_regression = st.checkbox('Ideale Regressionslinie hinzufügen', value=False)

    X = data[selected_x].values.reshape(-1, 1)
    y = data[selected_y].values
    model = None
    lowess_results = None

    if add_linear_regression:
        model = LinearRegression()
        model.fit(X, y)

    if add_ideal_regression:
        lowess_results = sm.nonparametric.lowess(y, X.squeeze(), frac=0.3)

    plot_data(X.squeeze(), y, model, lowess_results)
    st.pyplot(plt)
    plt.clf()

if __name__ == '__main__':
    main()
