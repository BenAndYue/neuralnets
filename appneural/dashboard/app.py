from os import read
import re
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
# from push import read_data
from dash.dependencies import Input, Output, State
from push import read_only, read_data
import base64
import io
import matplotlib.pyplot as plt
import numpy as np
import os
from dash.exceptions import PreventUpdate
import time
# read data in from database and only show 10 of them in tge dash datatable since this is stable for the process LOL
    # df_train = read_data()
    # dfObj1 = df_train.head(15)

# # put labels into y_train variable
# Y_train = df_train["label"]
# # Drop 'label' column
# X_train = df_train.drop(labels = ["label"],axis = 1) 
# # visualize number of digits classes
# count_status = str(Y_train.value_counts()) 

# dataframe used = df / count status next to the dashboard 
# to emphasize the object shown in the dataframe

def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

image_filename = 'plt.png'
app = dash.Dash(__name__)


#     html.Div([
#     html.Img(src='/assets/output.png'),
#     html.H1('Look at the test data'),
#     html.Div([
#         "Input: ",
#         dcc.Input(id='my-input', value='0.0-40.000', type='text')
#     ]),
#     html.Br(),
#     html.Div(id='my-output'),   
# ]),
# ])

app.layout = html.Div([
    dcc.Tabs(
        id="tabs-with-classes",
        value='tab-2',
        parent_className='custom-tabs',
        className='custom-tabs-container',
        children=[
            dcc.Tab(
                label='Tab one',
                value='tab-1',
                className='custom-tab',
                selected_className='custom-tab--selected'
            ),
            dcc.Tab(
                label='Tab two',
                value='tab-2',
                className='custom-tab',
                selected_className='custom-tab--selected'
            ),
            dcc.Tab(
                label='Tab three, multiline',
                value='tab-3', className='custom-tab',
                selected_className='custom-tab--selected'
            ),
            dcc.Tab(
                label='Tab four',
                value='tab-4',
                className='custom-tab',
                selected_className='custom-tab--selected'
            ),
        ]),
    html.Div(id='tabs-content-classes')
])
@app.callback(Output('tabs-content-classes', 'children'),
            Input('tabs-with-classes', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div(
        className="app-header",
        children=[
            html.Div('Digit Recognizer', className="app-header--title")]
    ), html.Div(
        children=html.Div([
            html.H1('Quick look to the data'),
            html.Div('''
                The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).
            ''')
        ])
    ),
    elif tab == 'tab-2':
        return html.Div([
    
    dcc.Input(id='input-1-state', type='number',max =29, min =-0),
    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
    html.Div(id='output-state'),
    # html.Img(src='/assets/plt.png')
    ])


    elif tab == 'tab-3':
        return html.Div([
            html.H3('Tab content 3')
        ])
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Tab content 4')
        ])

@app.callback(Output('output-state', 'children'),
                Input('submit-button-state', 'n_clicks'),
                State('input-1-state', 'value'), prevent_initial_call=True)
def update_output(n_clicks, input1):
    if n_clicks is None:
        raise PreventUpdate
    if input1== None:
        return html.Div("No input " + str(n_clicks) ) 
    index = int(input1)
    obj1= read_only(index)
    
    data = obj1
    data.drop(data.columns[[0,1]], axis = 1, inplace = True)
    sample_size = data.shape[0]
    validation_size = int(data.shape[0]*0.1)
    train_x = np.asarray(data.iloc[:sample_size-validation_size,1:]).reshape([sample_size-validation_size,28,28,1]) # taking all columns expect column 0
    plt.imshow(train_x[0].reshape([28,28]),cmap="Blues") 
    plt.axis("off")
    time.sleep(2)
    plt.savefig("plt.png")
    time.sleep(1)


    
    return  html.Div([
    html.Div(
        "You have selected: " + str(input1) + " ||||| Function used " + str(n_clicks)  + " amount of times|||||||||||||| RAW DATA = " 
    )
    # ])
    ,
    html.Img(src=b64_image(image_filename))])
    


    # return u'''
    #     Function has been used {} times,\n \n
    #     Index of one out of the database come out: "{}"
    #     id: "{}"
    #     data:image/png;base64,{}
    # '''.format(n_clicks, obj1,input1,dataa  )

if __name__ == '__main__':
    app.run_server(debug=True)