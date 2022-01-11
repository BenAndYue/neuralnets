from os import read
import re
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
# from push import read_data
from dash.dependencies import Input, Output, State
from push import read_only, read_data, read_test,read_set_test
import base64
import io
import matplotlib.pyplot as plt
import numpy as np
import os
from dash.exceptions import PreventUpdate
import time
import tensorflow as tf
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
image_filename2 = 'plt2.png'
image_filename3= 'plt3.png'
image_filename4= 'train.png'
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Tabs(
        id="tabs-with-classes",
        value='tab-2',
        parent_className='custom-tabs',
        className='custom-tabs-container',
        children=[
            dcc.Tab(
                label='Summary',
                value='tab-1',
                className='custom-tab',
                selected_className='custom-tab--selected'
            ),
            dcc.Tab(
                label='Training data',
                value='tab-2',
                className='custom-tab',
                selected_className='custom-tab--selected'
            ),
            dcc.Tab(
                label='Training matrix',
                value='tab-5', className='custom-tab',
                selected_className='custom-tab--selected'
            ),
            dcc.Tab(
                label='Keras DNN',
                value='tab-4',
                className='custom-tab',
                selected_className='custom-tab--selected'
            ),
            dcc.Tab(
                label='Keras CNN',
                value='tab-3',
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
    html.Div('Look at the train data. A small sample of the test data is saved of the because loading it would be to long.', className="app-header--title"),
    html.Br(),
    html.Div("Select a number between -1-30 to display the data of the test data."),
    dcc.Input(id='input-1-state', type='number',max =29, min =-0),
    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
    html.Div(id='output-state'),
    # html.Img(src='/assets/plt.png')
    ])

    elif tab == 'tab-3':
        return html.Div(
        className="app-header",
        children=[
            html.Div('Keras CNN', className="app-header--title"),
                html.Div('Select a number from 0-29 to pick a test document from the mongo database and run it throug the CNN model.'),
                html.Div(dcc.Input(id='input-on-submit', type='number',max =29, min =-0)),
                html.Button('Submit', id='submit-val', n_clicks=0),
                html.Div(id='container-button-basic',
                children='Enter a value and press submit')
]),
    elif tab == 'tab-4':
        return  html.Div(
        className="app-header",
        children=[
            html.Div('Keras DNN', className="app-header--title"),
            html.Div('Select a number from 0-29 to pick a test document from the mongo database and run it throug the DNN model.'),
                html.Div(dcc.Input(id='input-on-submit2', type='number',max =29, min =-0)),
                html.Button('Submit', id='submit-val2'),
                html.Div(id='container-button-basic2',
                children='Enter a value and press submit')
]),
    elif tab == 'tab-5':
        return html.Div([
            html.Div("Choose a number between -1 - 10 to display a matrix of numbers from the train dataset."),
            dcc.Input(id='my-id', type="number",max =10, min =-0),
    html.Button('Click Me', id='button'),
    html.Div(id='my-div')
])
# data training matrix
@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input('button', 'n_clicks')],
    state=[State(component_id='my-id', component_property='value')]
)
def update_output_div(n_clicks, input_value):
    if n_clicks is None:
        raise PreventUpdate
    if input_value== None:
        return html.Div("No input " + str(n_clicks) ) 
    index = int(input_value)
    obj1= read_set_test(index)
    
    data = obj1
    # get data
    label  = data['label'].iloc[0]
    label = str(label)

    # data prep
    data.drop(data.columns[[0,1]], axis = 1, inplace = True)
    sample_size = data.shape[0]
    validation_size = int(data.shape[0]*0.1)
    train_x = np.asarray(data.iloc[:sample_size-validation_size,1:]).reshape([sample_size-validation_size,28,28,1])
    rows = 1
    cols = obj1.shape[0]

    f = plt.figure(figsize=(2*cols,2*rows)) # defining a figure 

    for i in range(rows*cols): 
            f.add_subplot(rows,cols,i+1) # adding sub plot to figure on each iteration
            plt.imshow(train_x[i].reshape([28,28]),cmap="Blues") 
            plt.axis("off")
    plt.savefig("train.png")
    return html.Div([
        html.Div("You have selected: "+ str(index)),
        html.Img(src=b64_image(image_filename4))
        ])


# tab Keras DNN
@app.callback(
    Output('container-button-basic2', 'children'),
    Input('submit-val2', 'n_clicks'),
    State('input-on-submit2', 'value'))
def update_output(n_clicks, value):
    # if zero no reposne or no reload of the page
    if n_clicks is None:
        raise PreventUpdate
    if value== None:
        return html.Div("No input " + str(n_clicks) ) 
    # getting data from the database based on value given
    index = int(value)
    testdf = read_test(index)
    testdf.drop(testdf.columns[[0,1]], axis = 1, inplace = True)
    test_x = np.asarray(testdf.iloc[:,:]).reshape([-1,28,28,1])
    test_x = test_x/255
    # load model
    global model
    model = tf.keras.models.load_model('final_try2.h5',compile=False)

    test_y = np.argmax(model.predict(test_x),axis =1)
    plt.imshow(test_x[0].reshape([28,28]),cmap="Blues")
    plt.axis("off")
    plt.title("Predicted number:" + str(test_y[0]))
    plt.savefig("plt3.png")

    return html.Div([
        html.Div("You have selected: "+ str(index)),
        html.Img(src=b64_image(image_filename3))
        ])

# tab test data
@app.callback(Output('output-state', 'children'),
                Input('submit-button-state', 'n_clicks'),
                State('input-1-state', 'value'), prevent_initial_call=True)
def update_output2(n_clicks, input1):
    if n_clicks is None:
        raise PreventUpdate
    if input1== None:
        return html.Div("No input " + str(n_clicks) ) 
    index = int(input1)
    obj1= read_only(index)
    
    data = obj1
    # get data
    label  = data['label'].iloc[0]
    label = str(label)

    # data prep
    data.drop(data.columns[[0,1]], axis = 1, inplace = True)
    sample_size = data.shape[0]
    validation_size = int(data.shape[0]*0.1)
    train_x = np.asarray(data.iloc[:sample_size-validation_size,1:]).reshape([sample_size-validation_size,28,28,1]) # taking all columns expect column 0
    plt.imshow(train_x[0].reshape([28,28]),cmap="Blues") 
    plt.axis("off")
    # time.sleep(2)
    plt.savefig("plt.png")
    # time.sleep(2)
    return  html.Div([
        html.Div("You have selected index: " + str(input1) ),
        html.Br(),
        html.Div("Function used " + str(n_clicks) + " times" ),
        html.Br(),
        html.Div("Given number is:" + label  ),
        html.Img(src=b64_image(image_filename))])

# tab keras CNN
@app.callback(
    Output('container-button-basic', 'children'),
    Input('submit-val', 'n_clicks'),
    State('input-on-submit', 'value'))
def update_output(n_clicks, value):
    # if zero no reposne or no reload of the page
    if n_clicks is None:
        raise PreventUpdate
    if value== None:
        return html.Div("No input " + str(n_clicks) ) 
    # getting data from the database based on value given
    index = int(value)
    testdf = read_test(index)
    testdf.drop(testdf.columns[[0,1]], axis = 1, inplace = True)
    test_x = np.asarray(testdf.iloc[:,:]).reshape([-1,28,28,1])
    test_x = test_x/255
    # load model
    global model
    model = tf.keras.models.load_model('final_try.h5',compile=False)

    test_y = np.argmax(model.predict(test_x),axis =1)
    plt.imshow(test_x[0].reshape([28,28]),cmap="Blues")
    plt.axis("off")
    plt.title("Predicted number:" + str(test_y[0]))
    plt.savefig("plt2.png")
    # save img
    return html.Div([
        html.Div("You have selected: "+ str(index)),
        html.Img(src=b64_image(image_filename2))
        ])



if __name__ == '__main__':
    app.run_server(debug=True)