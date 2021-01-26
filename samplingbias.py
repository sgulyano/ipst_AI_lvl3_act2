import numpy as np
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

x = np.random.rand(100)*10 - 5
y = 5/(1 + np.exp(-x)) + np.random.randn(100)
# x = np.random.rand(100)
# y = np.random.rand(100)

x_min, x_max = x.min() - 0.1, x.max() + 0.1
y_min, y_max = y.min() - 0.1, y.max() + 0.1

f_input = go.Figure([go.Scatter(x=x, y=y, mode='markers')],
                    layout=go.Layout(
                        title='ข้อมูลทั้งหมด',
                        xaxis=dict(range=[x_min, x_max]),
                        xaxis_title='X',
                        yaxis=dict(range=[y_min, y_max]),
                        yaxis_title='Y',
))
f_input.layout.dragmode = 'lasso'
reg = LinearRegression().fit(x.reshape(x.size,1), y)
y_pred_best = reg.predict([[x_min], [x_max]])

mse_best = mean_squared_error(y, reg.predict(x.reshape(x.size,1)))

app.layout = html.Div(children=[
    html.H1(children='Sampling Bias'),

    html.Div(children='''
        ในกราฟทางซ้าย ให้ นักเรียน ลองเลือกจุดมาบางส่วน แล้วดูว่าผลลัพธ์ Linear Regression ที่ได้ ต่างจาก ถ้าใช้ข้อมูลทั้งหมดที่มีมากน้อยแค่ไหน
    '''),

    html.Div(children=[
        dcc.Graph(
            id='input-graph',
            figure=f_input
        )],
        style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}
    ),

    html.Div(children=[
        dcc.Graph(
            id='output-graph',
            figure=go.Figure()
        )],
        style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}
    ),

    html.Div([
        html.Div(id='accuracy-best-id', children=f'MSE = {mse_best:.3f}'),
        html.Div(id='accuracy-id')
    ],
    style={'textAlign': 'center'}
    )
])

@app.callback([Output(component_id='output-graph', component_property='figure'),
               Output(component_id='accuracy-id', component_property='children')], 
              [Input('input-graph', 'selectedData')])
def display_selected_data(selectedData):
    if selectedData and selectedData["points"]:
        idx = [row['pointIndex'] for row in selectedData["points"]]

        reg = LinearRegression().fit(x[idx].reshape(len(idx),1), y[idx])
        mse = mean_squared_error(y, reg.predict(x.reshape(x.size,1)))

        y_pred = reg.predict([[x_min], [x_max]])

        f_output = go.Figure([go.Scatter(x=x[idx], y=y[idx], mode='markers', name="Selected Data"), 
                              go.Scatter(x=[x_min, x_max], y=y_pred, mode='lines', name="Trend Line"),
                              go.Scatter(x=[x_min, x_max], y=y_pred_best, mode='lines', name="Best Line")],
                             layout=go.Layout(
            title='ข้อมูลที่ใช้ Train โมเดล',
            xaxis=dict(range=[x_min, x_max]),
            xaxis_title='X',
            yaxis=dict(range=[y_min, y_max]),
            yaxis_title='Y',
        ))
    else:
        f_output = go.Figure(layout=go.Layout(
            xaxis=dict(range=[x_min, x_max]),
            xaxis_title='X',
            yaxis=dict(range=[y_min, y_max]),
            yaxis_title='Y',
        ))
        mse = np.inf
    # print(json.dumps(selectedData, indent=2))
    return [f_output, f'Sampling Bias MSE = {mse:.3f}']


if __name__ == '__main__':
    app.run_server(debug=True)

