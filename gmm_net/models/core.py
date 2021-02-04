import numpy as np
import plotly.graph_objects as go
import dash_core_components as dcc


class Space():
    def __init__(self,
                 data: np.ndarray,
                 grid_points=None,
                 grid_mapping=None,
                 graph_whole=None,
                 graph_indv=None,
                 dropdown=None,
                 label_data=None,
                 label_feature=None,
                 is_middle_color_zero=None,
                 params_contour={},
                 params_scat_data={}):
        self.data = data.copy()
        self.n_data = data.shape[0]
        self.n_dim = data.shape[1]
        self.grid_points = grid_points
        self.grid_mapping = grid_mapping
        self.graph_whole = graph_whole
        self.graph_indiv = graph_indv
        self.dropdown = dropdown
        self.label_data = label_data
        self.label_feature = label_feature
        self.is_middle_color_zero = is_middle_color_zero
        self.dic_index_traces = {}
        self.params_contour = params_contour
        self.params_scat_data = params_scat_data
    
    def set_graph_whole(self, id, config=None):
        if config is None:
            config = {'displayModeBar': False}
        fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text='Latent space'),
                xaxis={
                    'range': [
                        self.data[:, 0].min() - 0.05,
                        self.data[:, 0].max() + 0.05
                    ]
                },
                yaxis={
                    'range': [
                        self.data[:, 1].min() - 0.05,
                        self.data[:, 1].max() + 0.05
                    ],
                    'scaleanchor': 'x',
                    'scaleratio': 1.0
                },
                showlegend=False
            )
        )
        # draw contour of mapping
        if self.is_middle_color_zero:
            zmid = 0.0
        else:
            zmid = None
        # if self.grid_mapping is not None:
        #     z = self.grid_mapping[:, 0]
        # else:
        #     z = None
        fig.add_trace(
            go.Contour(x=self.grid_points[:, 0],
                       y=self.grid_points[:, 1],
                       z=None,
                       name='contour',
                       zmid=zmid,
                       **self.params_contour
                       )
        )
        # draw invisible grids to click
        fig.add_trace(
            go.Scatter(x=self.grid_points[:, 0],
                       y=self.grid_points[:, 1],
                       mode='markers',
                       visible=True,
                       marker=dict(symbol='square', size=10, opacity=0.0, color='black'),
                       name='latent space')
        )
        self.dic_index_traces['grids'] = 1

        # draw latent variables
        fig.add_trace(
            go.Scatter(
                x=self.data[:, 0],
                y=self.data[:, 1],
                mode='markers',
                text=self.label_data,
                **self.params_scat_data
            )
        )
        self.dic_index_traces['data'] = 2

        # draw click point initialized by visible=False
        fig.add_trace(
            go.Scatter(
                x=np.array(0.0),
                y=np.array(0.0),
                visible=False,
                marker=dict(
                    size=12,
                    symbol='x',
                    color='#e377c2',
                    line=dict(
                        width=1.5,
                        color="white"
                    )
                ),
                name='clicked_point'
            )
        )
        self.dic_index_traces['clicked_point'] = 3

        self.graph_whole = dcc.Graph(
            id=id,
            figure=fig,
            config=config
        )

    def update_trace_clicked_point(self, clickData):
        if clickData['points'][0]['curveNumber'] == self.dic_index_traces['grids']:
            self.graph_whole.figure.update_traces(
                x=np.array(clickData['points'][0]['x']),
                y=np.array(clickData['points'][0]['y']),
                visible=True,
                marker=dict(
                    symbol='x'
                ),
                selector=dict(name='clicked_point', type='scatter')
            )
        elif clickData['points'][0]['curveNumber'] == self.dic_index_traces['data']:
            self.graph_whole.figure.update_traces(
                x=np.array(clickData['points'][0]['x']),
                y=np.array(clickData['points'][0]['y']),
                visible=True,
                marker=dict(
                    symbol='circle-x'
                ),
                selector=dict(name='clicked_point', type='scatter')
            )
            # if latent variable is clicked
            # fig_ls.update_traces(visible=False, selector=dict(name='clicked_point'))

    # def _set_grid(self, grid_points, n_grid_points):
    #     self.grid_points = grid_points

class LatentSpace(Space):
    pass

class ObservedSpace(Space):
    pass

class FeatureSpace(Space):
    pass
