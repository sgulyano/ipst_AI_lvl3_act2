import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import HoverTool, ColumnDataSource, Div, Paragraph
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, factor_mark
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


## generate data
x = np.random.rand(100)*10 - 5
y = 5/(1 + np.exp(-x)) + np.random.randn(100)

# get min/max data
x_min, x_max = x.min() - 0.1, x.max() + 0.1
y_min, y_max = y.min() - 0.1, y.max() + 0.1

# train best linear regression
reg = LinearRegression().fit(x.reshape(x.size,1), y)
y_pred_best = reg.predict([[x_min], [x_max]])

mse_best = mean_squared_error(y, reg.predict(x.reshape(x.size,1)))
mse_selected = np.inf

# get figures
TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "(@x, @y)"),
]

s1 = ColumnDataSource(data=dict(x=x, y=y))
p1 = figure(plot_width=400, plot_height=400, 
            x_range=(x_min, x_max), y_range=(y_min, y_max), 
            tooltips=TOOLTIPS,
            x_axis_label='X',
            y_axis_label='Y',
            tools="lasso_select,crosshair,reset,save,wheel_zoom", title="เลือกจุดโดยการคลิกค้างไว้ แล้วลากล้อมจุดที่ต้องการ")
p1.scatter('x', 'y', source=s1, alpha=0.4, size=8)

s2 = ColumnDataSource(data=dict(x=[], y=[]))
p2 = figure(plot_width=400, plot_height=400, 
            x_range=(x_min, x_max), y_range=(y_min, y_max),
            x_axis_label='X',
            y_axis_label='Y',
            tools="crosshair,reset,save,wheel_zoom", title="ข้อมูลที่ใช้สร้างโมเดล")

p2.scatter('x', 'y', source=s2, alpha=0.4, size=8)
p2.line([x_min, x_max], y_pred_best, line_width=2, color='green',
        legend_label='โมเดลจากข้อมูลทั้งหมด')

s2_reg = ColumnDataSource(data=dict(x=[], y=[]))
p2.line('x', 'y', source=s2_reg, line_width=2, color='orange',
        legend_label='โมเดลที่ได้')


# Set up dashboard
title = Div(text="""<H1>อคติในการเลือกตัวอย่าง (Sampling Bias)</H1>""")
desc = Paragraph(text="""ในกราฟทางซ้าย ให้ นักเรียน ลองเลือกจุดมาบางส่วน แล้วดูว่าผลลัพธ์ Linear Regression ที่ได้ ต่างจาก ถ้าใช้ข้อมูลทั้งหมดที่มีมากน้อยแค่ไหน""")
header = column(title, desc, sizing_mode="scale_both")

# graph and mse results
graphs = row(p1, p2, width=800)

mse_best_txt = Div(text=f'\t\tMSE ของ โมเดลจากข้อมูลทั้งหมด = {mse_best:.3f}')
mse_selected_txt = Div(text=f'\t\tMSE ของ โมเดลที่ได้ = {mse_selected:.3f}')
body = column(graphs, mse_best_txt, mse_selected_txt)

# callbacks
def update_data(attrname, old, new):
    d1 = s1.data
    idx = s1.selected.indices
    x_new = d1['x'][idx]
    y_new = d1['y'][idx]
    s2.data = dict(x=x_new, y=y_new)

    if len(x_new) > 0 and len(y_new) > 0:
        reg = LinearRegression().fit(x_new.reshape(x_new.size,1), y_new)
        y_pred_best = reg.predict([[x_min], [x_max]])
        s2_reg.data = dict(x=[x_min, x_max], y=y_pred_best)

        mse_selected = mean_squared_error(y, reg.predict(x.reshape(x.size,1)))
        mse_selected_txt.text = f'\t\tMSE ของ โมเดลที่ได้ = {mse_selected:.3f}'

        p2.legend.location = "top_left"
        p2.legend.click_policy="hide"
    else:
        s2_reg.data = dict(x=[], y=[])
    
s1.selected.on_change('indices', update_data)

#
curdoc().add_root(column(header, body))
curdoc().title = "อคติในการเลือกตัวอย่าง"