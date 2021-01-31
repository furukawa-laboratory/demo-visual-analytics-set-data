import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go

fig=go.Figure()
graph=dcc.Graph(
    id='test_graph',
    figure=fig,
    config=None
)


pass