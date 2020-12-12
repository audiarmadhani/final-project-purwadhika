import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pandas as pd

from app import app

df = pd.read_csv('/Users/audia/OneDrive/Documents/Python Scripts/dashboard/Final Project/Dashboard/dash_data_new_2.csv')
df['dest_airport'] = df['dest_airport'].fillna('airport')

layout = layout = html.Div([
    html.Div([
        dash_table.DataTable(
            id='table'
            ,style_cell={
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'maxWidth': '50px',
                'backgroundColor': 'rgb(50, 50, 50)',
                'color': 'white',
            }
            , style_table={
                'paddingLeft': 50,
                'paddingRight': 50,
                'paddingTop': 100,
                'overflowY': 'scroll',
            }
            ,columns=[{"name": i, "id": i} for i in df.columns]
            ,fixed_rows={ 'headers': True, 'data': 0 }
            ,data=df.to_dict('records')
        )
    ]),

    dcc.Link(
        'Return to Dashboard', 
        href='app1',
        style={
            'paddingLeft' : 50,
            'paddingTop': 100,
            }),

])