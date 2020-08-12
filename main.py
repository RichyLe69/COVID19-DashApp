import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
from datetime import datetime
from os.path import isfile

baseURL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
fileNamePickle = "allData.pkl"

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

tickFont = {'size': 12, 'color': "rgb(30,30,30)", 'family': "Courier New, monospace"}


def load_data_global(file_name, column_name):
    agg_dict = {column_name: sum, 'Lat': np.median, 'Long': np.median}
    data = pd.read_csv(baseURL + file_name) \
        .rename(columns={'Country/Region': 'Country'}) \
        .melt(id_vars=['Country', 'Province/State', 'Lat', 'Long'], var_name='date', value_name=column_name) \
        .astype({'date': 'datetime64[ns]', column_name: 'Int64'}, errors='ignore')
    # Extract chinese provinces separately.
    data_china = data[data.Country == 'China']
    data = data.groupby(['Country', 'date']).agg(agg_dict).reset_index()
    data['Province/State'] = '<all>'
    return pd.concat([data, data_china])


def load_data_us(file_name, column_name):
    id_vars = ['Country', 'Province/State', 'Lat', 'Long']
    agg_dict = {column_name: sum, 'Lat': np.median, 'Long': np.median}
    data = data = pd.read_csv(baseURL + file_name).iloc[:, 6:]
    if 'Population' in data.columns:
        data = data.drop('Population', axis=1)
    data = data \
        .drop('Combined_Key', axis=1) \
        .rename(columns={'Country_Region': 'Country', 'Province_State': 'Province/State', 'Long_': 'Long'}) \
        .melt(id_vars=id_vars, var_name='date', value_name=column_name) \
        .astype({'date': 'datetime64[ns]', column_name: 'Int64'}, errors='ignore') \
        .groupby(['Country', 'Province/State', 'date']).agg(agg_dict).reset_index()
    return data


def simple_moving_average(df, length=7):
    return df.rolling(length).mean()


def refresh_data():
    data_global = load_data_global("time_series_covid19_confirmed_global.csv", "CumConfirmed") \
        .merge(load_data_global("time_series_covid19_deaths_global.csv", "CumDeaths"))
    data_us = load_data_us("time_series_covid19_confirmed_US.csv", "CumConfirmed") \
        .merge(load_data_us("time_series_covid19_deaths_US.csv", "CumDeaths"))
    data = pd.concat([data_global, data_us])
    data.to_pickle(fileNamePickle)
    return data


def all_data():
    if not isfile(fileNamePickle):
        refresh_data()
    full_data = pd.read_pickle(fileNamePickle)
    return full_data


countries = all_data()['Country'].unique()
countries.sort()

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# App title, keywords and tracking tag (optional).
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=UA-161733256-2"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());
          gtag('config', 'UA-161733256-2');
        </script>
        <meta name="keywords" content="COVID-19,Coronavirus,Dash,Python,Dashboard,Cases,Statistics">
        <title>COVID-19 Case History</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
       </footer>
    </body>
</html>"""

app.layout = html.Div(
    style={'font-family': "Courier New, monospace"},
    children=[
        html.H1('Case History of the Coronavirus (COVID-19)'),
        html.Div(className="row", children=[
            html.Div(className="four columns", children=[
                html.H5('Country'),
                dcc.Dropdown(
                    id='country',
                    options=[{'label': c, 'value': c} for c in countries],
                    value='US'
                )
            ]),
            html.Div(className="four columns", children=[
                html.H5('State / Province'),
                dcc.Dropdown(
                    id='state'
                )
            ]),
            html.Div(className="four columns", children=[
                html.H5('Selected Metrics'),
                dcc.Checklist(
                    id='metrics',
                    options=[{'label': m, 'value': m} for m in ['Confirmed', 'Deaths']],
                    value=['Confirmed', 'Deaths']
                )
            ])
        ]),
        dcc.Graph(
            id="plot_new_metrics",
            config={'displayModeBar': False}
        ),
        dcc.Graph(
            id="plot_cum_metrics",
            config={'displayModeBar': False}
        ),
        dcc.Interval(
            id='interval-component',
            interval=3600 * 1000,  # Refresh data each hour.
            n_intervals=0
        )
    ]
)


@app.callback(
    [Output('state', 'options'), Output('state', 'value')],
    [Input('country', 'value')]
)
def update_states(country):
    d = all_data()
    states = list(d.loc[d['Country'] == country]['Province/State'].unique())
    states.insert(0, '<all>')
    states.sort()
    state_options = [{'label': s, 'value': s} for s in states]
    state_value = state_options[0]['value']
    return state_options, state_value


def filtered_data(country, state):
    d = all_data()
    data = d.loc[d['Country'] == country].drop('Country', axis=1)
    if state == '<all>':
        data = data.drop('Province/State', axis=1).groupby("date").sum().reset_index()
    else:
        data = data.loc[data['Province/State'] == state]
    new_cases = data.select_dtypes(include='Int64').diff().fillna(0)
    new_cases.columns = [column.replace('Cum', 'New') for column in new_cases.columns]
    data = data.join(new_cases)
    data['dateStr'] = data['date'].dt.strftime('%b %d, %Y')
    data['NewDeathsSMA7'] = simple_moving_average(data.NewDeaths, length=7)
    data['NewConfirmedSMA7'] = simple_moving_average(data.NewConfirmed, length=7)
    return data


def add_trend_lines(figure, data, metrics, prefix):
    if prefix == 'New':
        for metric in metrics:
            figure.add_trace(
                go.Scatter(
                    x=data.date, y=data[prefix + metric + 'SMA7'],
                    mode='lines', line=dict(
                        width=3, color='rgb(200,30,30)' if metric == 'Deaths' else 'rgb(100,140,240)'
                    ),
                    name='Rolling 7-Day Average of Deaths' if metric == 'Deaths' \
                        else 'Rolling 7-Day Average of Confirmed'
                )
            )


def barchart(data, metrics, prefix="", yaxis_title=""):
    figure = go.Figure(data=[
        go.Bar(
            name=metric, x=data.date, y=data[prefix + metric],
            marker_line_color='rgb(0,0,0)', marker_line_width=1,
            marker_color={'Deaths': 'rgb(200,30,30)', 'Confirmed': 'rgb(100,140,240)'}[metric]
        ) for metric in metrics
    ])
    add_trend_lines(figure, data, metrics, prefix)
    figure.update_layout(
        barmode='group', legend=dict(x=.05, y=0.95, font={'size': 15}, bgcolor='rgba(240,240,240,0.5)'),
        plot_bgcolor='#FFFFFF', font=tickFont) \
        .update_xaxes(
        title="", tickangle=-90, type='category', showgrid=True, gridcolor='#DDDDDD',
        tickfont=tickFont, ticktext=data.dateStr, tickvals=data.date) \
        .update_yaxes(
        title=yaxis_title, showgrid=True, gridcolor='#DDDDDD')
    return figure


@app.callback(
    [Output('plot_new_metrics', 'figure'), Output('plot_cum_metrics', 'figure')],
    [Input('country', 'value'), Input('state', 'value'), Input('metrics', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_plots(country, state, metrics, n):
    refresh_data()
    data = filtered_data(country, state)
    barchart_new = barchart(data, metrics, prefix="New", yaxis_title="New Cases per Day")
    barchart_cum = barchart(data, metrics, prefix="Cum", yaxis_title="Cumulated Cases")
    return barchart_new, barchart_cum


server = app.server

if __name__ == '__main__':
    app.run_server(host="0.0.0.0")

# http://127.0.0.1: with given port number
