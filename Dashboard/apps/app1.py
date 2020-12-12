import dash
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_table
from dash.dependencies import Input, Output
import flask
import os
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import matplotlib.pyplot as plt
import matplotlib as mpl
import requests
import io
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import pmdarima as pm
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

from datetime import datetime
from tqdm import tqdm, tqdm_notebook
from numpy.linalg import LinAlgError

from app import app

df = pd.read_csv('/Users/audia/OneDrive/Documents/Python Scripts/dashboard/Final Project/Dashboard/dash_data_new_2.csv')
df['dest_airport'] = df['dest_airport'].fillna('airport')


mapbox_access_token = 'pk.eyJ1IjoiYXVkaWFybWFkaGFuaSIsImEiOiJja2k5c3A2MmowaXIyMnFyb2Y5amhjOTdvIn0.ftN2zUjIC4j26jtOOQc2EQ'

layout = html.Div([
    html.H1(
        'Aircraft Payload Dashboard',
        style={
            'paddingLeft' : 50,
            }
        ),
    html.Div([   # Holds the widgets & Descriptions

        html.Div([  

            html.Div(
                '''You can explore the payload on selected airport.''',
                style={
                }
            ),
            html.Div(
                '''Select airport:''',
                style={
                    'paddingTop' : 20,
                    'paddingBottom' : 10,
                }
            ),
            dcc.Dropdown(
                options=[{'label': x, 'value': x}
                     for x in df["dest_airport"].unique().tolist()],
                value="Soekarno-Hatta International Airport",
                multi=False,
                id="dest_airport_dropdown",
                style={
                }
                
            ),
         
        ],
        style={
            "width" : '30%', 
            'display' : 'inline-block', 
            'paddingLeft' : 50, 
            'paddingRight' : 10,
            'boxSizing' : 'border-box',
            }
        ),
        
        html.Div([  # Holds the map & the widgets

            dcc.Graph(id="machine_learning") # Holds the map in a div to apply styling to it
            
        ],
        style={
            "width" : '70%', 
            'float' : 'right', 
            'display' : 'inline-block', 
            'paddingRight' : 50, 
            'paddingLeft' : 10,
            'boxSizing' : 'border-box',
            })

    ],
    style={'paddingBottom' : 20}),

    html.Div([  # Holds the map, barchart & piechart (40:30:30 split) 
        html.Div([
            dcc.Graph(
                id="payload_map",
            ),
        ],
        style={
            "width" : '40%', 
            'float' : 'left', 
            'display' : 'inline-block', 
            'paddingRight' : 5, 
            'paddingLeft' : 50,
            'boxSizing' : 'border-box'
            }
        ),
        html.Div([
            dcc.Graph(
                id="monthly_payload_bar",
            ),
        ],
        style={
            "width" : '30%', 
            'float' : 'left', 
            'display' : 'inline-block', 
            'paddingRight' : 5, 
            'paddingLeft' : 50,
            'boxSizing' : 'border-box'
            }
        ),
        html.Div([
            dcc.Graph(
                id="payload_type_pie",
            )
            #style={'height' : '50%'})
        ],
        style={
            "width" : '30%', 
            'float' : 'right', 
            'display' : 'inline-block', 
            'paddingRight' : 50, 
            'paddingLeft' : 5,
            'boxSizing' : 'border-box'
            })

    ]),
    
    dcc.Link(
        'See full table', 
        href='app2',
        style={
            'paddingLeft' : 50,
            }),

    html.Div([
        # Add footer
        html.Div(
            'Audi Armadhani - 2020',
            style={
                # 'fontFamily' : font_family,
                'fontSize' : 10,
                'fontStyle' : 'italic',
                'paddingTop' : 50,
                'paddingLeft' : 50,
                'paddingBottom' : 10,
                }
            ),
        html.Div(
            'for purwadhika final project',
            style={
                # 'fontFamily' : font_family,
                'fontSize' : 10,
                'fontStyle' : 'italic',
                'paddingLeft' : 50,
                'paddingBottom' : 10,
                }
            )
        ])
])



@app.callback(Output('payload_map', 'figure'),
              [Input('dest_airport_dropdown', 'value')])
def create_payload_map(destination_airport):
    dff = df[df['dest_airport']==destination_airport].groupby('orig_airport', as_index=False).agg({'payload':'sum','orig_lat':'first','orig_long':'first'}).sort_values('payload', ascending=False)
    
    trace = [{
                'type' : 'scattermapbox',
                'mode' : 'markers',
                'customdata' : dff["payload"],
                'lat' : dff["orig_lat"],
                'lon' : dff["orig_long"],
                'marker' : {
                    'color' : dff["payload"],
                    'size' : dff["payload"]/5000,
                    },
                'text' : dff["orig_airport"],
                'hovertemplate':
                "<b>%{text}</br></br></br>"+
                "Payload: %{customdata} kg"
                ,
            }]
            
    layout = {
        'height' : 300,
        'paper_bgcolor' : 'rgb(0,0,0)',
              'font' : {
                  'color' : 'rgb(250,250,250'
              }, # Set this to match the colour of the sea in the mapbox colourscheme
        'autosize' : True,
        'hovermode' : 'closest',
        'mapbox' : {
            'accesstoken' : mapbox_access_token,
            'center' : {
                'lat' : 0.7893,
                'lon' : 113.9213
            },
            'zoom' : 3,
            'style' : 'dark',   # Dark theme will make the colours stand out
        },
        'margin' : {'t' : 0,
                   'b' : 0,
                   'l' : 0,
                   'r' : 0},
        'legend' : {
            'font' : {'color' : 'white'},
             'orientation' : 'h',
             'x' : 0,
             'y' : 1.01
        }
    }
    fig = dict(data=trace, layout=layout) 
    return fig

@app.callback(Output('monthly_payload_bar', 'figure'),
              [Input('dest_airport_dropdown', 'value')])
def create_payload_map(destination_airport):
    dff = df[df['dest_airport']==destination_airport].groupby('month_name', as_index=False, sort=False).sum()

    trace = [{
            'type' : 'scatter',
            'y' : dff['payload'],
            'x' : dff['month_name'],
            'hoverinfo' : dff['payload'],
            'marker' : {
            'line' : {'width' : 2}},
        }]
    
    fig = {'data' : trace,
          'layout' : {
              'paper_bgcolor' : 'rgb(0,0,0)',
              'plot_bgcolor' : 'rgb(0,0,0)',
              'font' : {
                  'color' : 'rgb(250,250,250)'
              },
              'height' : 300,
              'title' : 'test',
              'margin' : { # Set margins to allow maximum space for the chart
                  'b' : 25,
                  'l' : 30,
                  't' : 70,
                  'r' : 0
              },
              'legend' : { # Horizontal legens, positioned at the bottom to allow maximum space for the chart
                  'orientation' : 'h',
                  'x' : 0,
                  'y' : 1.01,
                  'yanchor' : 'bottom',
                  },
                }
          }
    
    # Returns the figure into the 'figure' component property, update the bar chart
    return fig


@app.callback(Output('payload_type_pie', 'figure'),
              [Input('dest_airport_dropdown', 'value')])
def create_piechart(destination_airport):
    dff = df[df['dest_airport']==destination_airport].groupby('commodity', as_index=False, sort=False).sum()

    piechart=px.pie(dff,
        values=dff.payload,
        names=dff.commodity,
        color_discrete_sequence=px.colors.sequential.Teal,
    )

    piechart.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'font_color':'rgb(200,200,200)',
        'font_size':11,
        'height':300,
        'margin':{'l':0}
    })

    piechart.update_traces(textposition='inside')


    return (piechart)


@app.callback(Output('machine_learning', 'figure'),
              [Input('dest_airport_dropdown', 'value')])

def modellingAudi(origin_airport):
    
    #holiday dataset
    df_holiday = pd.read_csv('/Users/audia/OneDrive/Documents/Python Scripts/dashboard/Final Project/Dashboard/holiday.csv')
    df_holiday = df_holiday.drop(['Unnamed: 0'], axis=1)
    df_holiday['date'] = pd.to_datetime(df_holiday['date'])
    df_holiday = df_holiday.groupby('date', as_index=True).sum()
    
    #split date
    split_date ='2019-10-01'
    df_holiday_training = df_holiday.loc[df_holiday.index <= split_date]
    df_holiday_test = df_holiday.loc[df_holiday.index > split_date]
    
    #create empty time series data set
    date_rng = pd.date_range(start='1/1/2019', end='12/31/2019', freq='D')
    date_rng = pd.DataFrame(date_rng).rename(columns={0:'flown_date'}).set_index('flown_date')

    #import dataframe
    df = pd.read_csv('/Users/audia/OneDrive/Documents/Python Scripts/dashboard/Final Project/Dashboard/dash_data_new_2.csv', parse_dates=['flown_date'])
    
    #filter based on origin airport
    df = df[df['orig_airport'] == origin_airport]
    
    #take flown date and payload
    df_tmp = df[['flown_date','payload']].copy().groupby('flown_date', as_index=True).sum()
    
    # #imputing for NaN values
    df_merged = pd.concat([date_rng, df_tmp], axis=1)
    df_tmp_2 = df_merged.assign(InterpolateTime=df_merged.payload.fillna(df_merged.payload.interpolate(method='time'))).drop('payload', axis=1).rename(columns={'InterpolateTime':'payload'})  

    #train test split
    df_training = df_tmp_2.loc[df_tmp_2.index <= split_date]
    df_test = df_tmp_2.loc[df_tmp_2.index > split_date]

    df_predict = pd.read_csv('/Users/audia/OneDrive/Documents/Python Scripts/Data Analysis/FINAL PROJECT/model_results/{}_predict.csv'.format(origin_airport))
    df_predict['flown_date'] = pd.to_datetime(df_predict['flown_date'])
    df_predict = df_predict.groupby('flown_date', as_index=True).sum()

    df_forecast = pd.read_csv('/Users/audia/OneDrive/Documents/Python Scripts/Data Analysis/FINAL PROJECT/model_results/{}_forecast.csv'.format(origin_airport))
    df_forecast['Unnamed: 0'] = pd.to_datetime(df_forecast['Unnamed: 0'])
    df_forecast = df_forecast.rename(columns={'Unnamed: 0':'flown_date'}).groupby('flown_date', as_index=True).sum()


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_training.index, y=df_training.payload, mode='lines', name='Training Dataset'))
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test.payload, mode='lines', name='Test Dataset'))
    fig.add_trace(go.Scatter(x=df_predict.index, y=df_predict.payload, mode='lines', name='Prediction Dataset'))
    fig.add_trace(go.Scatter(x=df_forecast.index, y=df_forecast.payload, mode='lines', name='Forecast Dataset'))
    
    fig.update_layout(
        paper_bgcolor='rgb(0,0,0)',
        plot_bgcolor='rgb(0,0,0)',
        font_color='rgb(250,250,250)',
        )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return (fig)

