import streamlit as st
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def f(x):
    return .5 * x + np.sqrt(np.max(x, 0)) - np.cos(x) + 2

def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def generate_data_and_plot(N, sigma_epsilon, poly, index):
    x_max = 3.0
    x = x_max * (2 * np.random.rand(N) - 1)
    epsilon = sigma_epsilon * np.random.randn(N)
    y = f(x) + epsilon
    
    x_range = np.linspace(-x_max, 3.2, 1000)
    function_line = f(x_range)
    
    p1 = np.poly1d(np.polyfit(x, y, 1))
    p5 = np.poly1d(np.polyfit(x, y, poly))
    
    mse1 = calculate_mse(y, p1(x))
    mse5 = calculate_mse(y, p5(x))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='grey', size=5), name='Datenpunkte'))
    fig.add_trace(go.Scatter(x=x_range, y=function_line, mode='lines', name='Wahrer Zusammenhang', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=[3.2], y=[f(3.2)], mode='markers', marker=dict(color='red', size=10), name=f"f(3.2)={f(3.2):.2f}"))
    fig.add_trace(go.Scatter(x=x_range, y=p1(x_range), mode='lines', name='Einfaches Modell', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=[3.2], y=[p1(3.2)], mode='markers', marker=dict(color='blue', size=10), name=f"Einfaches Modell f^(3.2)={p1(3.2):.2f}"))
    fig.add_trace(go.Scatter(x=x_range, y=p5(x_range), mode='lines', name='Komplexeres Modell', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=[3.2], y=[p5(3.2)], mode='markers', marker=dict(color='green', size=10), name=f"Komplexeres Modell f^(3.2)={p5(3.2):.2f}"))
    fig.add_trace(go.Scatter(x=[3.2], y=[0], mode='markers', marker=dict(color='black', size=5), name='x=3.2'))
    fig.update_layout(
        title=f"Stichprobe {index + 1}",
        annotations=[
            dict(
                x=1.05,
                y=0.05,
                xref="paper",
                yref="paper",
                text=f"MSE Einfaches Modell: {mse1:.2f}, MSE Komplexeres Modell: {mse5:.2f},",
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
        ]
    )
    return fig

def simulate_resampling(x_max, sigma_epsilon, f, poly, x_test=3.2, R=1000, N=100):
    y_hat_linear = np.zeros(R)
    y_hat_poly = np.zeros(R)
    
    for r in range(R):
        x = x_max * (2 * np.random.rand(N) - 1)
        epsilon = sigma_epsilon * np.random.randn(N)
        y = f(x) + epsilon

        p_linear = np.poly1d(np.polyfit(x, y, 1))
        y_hat_linear[r] = p_linear(x_test)
        
        p_poly = np.poly1d(np.polyfit(x, y, poly))
        y_hat_poly[r] = p_poly(x_test)
    
    return y_hat_linear, y_hat_poly

def plot_histograms(y_hat_linear, y_hat_poly, x_test, f_x_test):
    fig = make_subplots(rows=2, cols=1)
    
    fig.add_trace(go.Histogram(x=y_hat_linear, name='Einfaches Modell', opacity=0.6), row=1, col=1)
    
    fig.add_trace(go.Histogram(x=y_hat_poly, name='Komplexeres Modell', opacity=0.6), row=2, col=1)
    
    for i in range(1, 3):
        fig.add_shape(type='line',
                      x0=f_x_test, y0=0,
                      x1=f_x_test, y1=100,
                      line=dict(color="Red"),
                      row=i, col=1)
    
    # Schwarze Linie für den Mittelwert der Vorhersagen
    fig.add_shape(type='line',
                  x0=np.mean(y_hat_linear), y0=0,
                  x1=np.mean(y_hat_linear), y1=100,
                  line=dict(color="Black"),
                  row=1, col=1)
    fig.add_shape(type='line',
                  x0=np.mean(y_hat_poly), y0=0,
                  x1=np.mean(y_hat_poly), y1=100,
                  line=dict(color="Black"),
                  row=2, col=1)
    
    fig.update_layout(title_text="Histogramme der Vorhersagen für f(3.2)")
    fig.update_xaxes(title_text="Vorhersage für 1000 Stichproben einfaches Modell f^(3.2)", row=1, col=1)
    fig.update_xaxes(title_text="Vorhersage für 1000 Stichproben komplexeres Modell f^(3.2)", row=2, col=1)
    fig.update_yaxes(title_text="Häufigkeit", row=1, col=1)
    fig.update_yaxes(title_text="Häufigkeit", row=2, col=1)
    
    return fig


def home_page():
    st.title("Eine wahrer Zusammenhang und seine Daten")
    st.write("Normalerweise kennen wir den wahren Zusammenhang nicht. Hier in dem Beispiel schon. Er ist:")
    st.latex(r"y = 0.5x + \sqrt{\max(x, 0)} - \cos(x) + 2")
    
    N = st.slider('Anzahl Datenpunkte (N)', min_value=10, max_value=1000, value=500, step=10)
    sigma_epsilon = st.slider('Noise (sigma_epsilon)', min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    poly = st.slider('Komplexität des komplexerem Modells (Grad des Polynoms)', min_value=1, max_value=10, value=5, step=1, key='poly2')
    
    x_max = 3
    x = x_max * (2 * np.random.rand(N) - 1)
    epsilon = sigma_epsilon * np.random.randn(N)
    y = f(x) + epsilon
    
    x_range = np.linspace(-x_max, x_max, 1000)
    function_line = f(x_range)
    
    p1 = np.poly1d(np.polyfit(x, y, 1)) 
    p5 = np.poly1d(np.polyfit(x, y, poly))  
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='grey', size=5), name='Datenpunkte'))
    
    # Checkbox for True Function
    if st.checkbox("Zeige wahren Zusammenhang", value=True):
        fig.add_trace(go.Scatter(x=x_range, y=function_line, mode='lines', name='Wahrer Zusammenhang', line=dict(color='red')))
    
    if st.checkbox("Zeige einfaches Modell"):
        fig.add_trace(go.Scatter(x=x_range, y=p1(x_range), mode='lines', name='Einfaches Modell', line=dict(color='blue')))
    
    if st.checkbox("Zeige komplexeres Modell"):
        fig.add_trace(go.Scatter(x=x_range, y=p5(x_range), mode='lines', name='Komplexeres Modell', line=dict(color='green')))
    
    st.plotly_chart(fig, use_container_width=True)

def page_2():
    N = st.slider('Anzahl Datenpunkte (N)', min_value=10, max_value=100, value=20, step=1, key='N2')
    sigma_epsilon = st.slider('Noise (sigma_epsilon)', min_value=0.1, max_value=5.0, value=1.0, step=0.1, key='sigma2')
    poly = st.slider('Komplexität des komplexerem Modells (Grad des Polynoms)', min_value=1, max_value=10, value=5, step=1, key='poly2')
    
    x_max = 3.0
    x = x_max * (2 * np.random.rand(N) - 1)
    epsilon = sigma_epsilon * np.random.randn(N)
    y = f(x) + epsilon
    
    def display_plots():
        if 'graphs' in st.session_state:
            for fig in st.session_state.graphs:
                st.plotly_chart(fig, use_container_width=True)
    
    def display_histogram():
        if 'histogram' in st.session_state:
            st.plotly_chart(st.session_state.histogram, use_container_width=True)
    
    if st.button('Fünf neue Stichproben ziehen'):
        st.session_state.graphs = [generate_data_and_plot(N, sigma_epsilon, poly, i) for i in range(5)]
        y_hat_linear, y_hat_poly = simulate_resampling(x_max, sigma_epsilon, f, poly, x_test=3.2, R=1000, N=100)
        st.session_state.histogram = plot_histograms(y_hat_linear, y_hat_poly, 3.2, f(3.2))
        display_plots()
        display_histogram()
        
    if st.button('Clear All Plots'):
        if 'graphs' in st.session_state:
            del st.session_state['graphs']
        if 'histogram' in st.session_state:
            del st.session_state['histogram']
        st.experimental_rerun()  
    
    display_plots()
    
    st.write("Wenn wir nun 1000 Stichproben ziehen:")
    display_histogram()

def main():
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Auswahl", list(pages.keys()))
    
    page = pages[selection]
    with st.spinner(f"Lade Seite: {selection}"):
        page()

pages = {
    "Wahres Modell und Daten": home_page,
    "Experiment: Wir ziehen Stichproben": page_2  
}

if __name__ == "__main__":
    main()
