import numpy as np
import plotly.graph_objects as go
import dash_core_components as dcc
from scipy.spatial.distance import cdist
import dash


class Space():
    def __init__(self,
                 data: np.ndarray,
                 grid_points=None,
                 grid_mapping=None,
                 graph_whole=None,
                 graph_indv=None,
                 store_fig_whole=None,
                 store_fig_indv=None,
                 dropdown=None,
                 label_data=None,
                 label_feature=None,
                 is_middle_color_zero=None,
                 params_contour={},
                 params_scat_data={},
                 params_figure_layout={}):
        self.data = data.copy()
        self.n_data = data.shape[0]
        self.n_dim = data.shape[1]
        self.grid_points = grid_points
        self.grid_mapping = grid_mapping
        self.graph_whole = graph_whole
        self.graph_indiv = graph_indv
        self.store_fig_whole = store_fig_whole
        self.store_fig_indiv = store_fig_indv
        self.dropdown = dropdown
        self.label_data = label_data
        self.label_feature = label_feature
        self.is_middle_color_zero = is_middle_color_zero
        self.dic_index_traces = {}
        self.params_contour = params_contour
        self.params_scat_data = params_scat_data
        self.params_figure_layout = params_figure_layout
        self.index_clicked_grid = None

    def set_graph_whole(self, id_graph, id_store, annotation_text='', config=None):
        if config is None:
            config = {'displayModeBar': False}
        x_range = [
            self.data[:, 0].min(),
            self.data[:, 0].max()
        ]
        y_range = [
            self.data[:, 1].min(),
            self.data[:, 1].max()
        ]
        fig = go.Figure(
            layout=go.Layout(
                xaxis={
                    'range': [self.data[:, 0].min() - 0.1, self.data[:, 0].max() + 0.1],
                    'visible': False
                },
                yaxis={
                    'range': [self.data[:, 1].min() - 0.1, self.data[:, 1].max() + 0.1],
                    'visible': False,
                    'scaleanchor': 'x',
                    'scaleratio': 1.0
                },
                showlegend=False,
                **self.params_figure_layout
            )
        )
        # draw contour of mapping
        if self.is_middle_color_zero:
            self.params_contour.update(dict(zmid=0.0))
        else:
            self.params_contour.update(dict(zmid=None))
        # if self.grid_mapping is not None:
        #     z = self.grid_mapping[:, 0]
        # else:
        #     z = None
        fig.add_trace(
            go.Contour(x=self.grid_points[:, 0],
                       y=self.grid_points[:, 1],
                       z=None,
                       name='contour',
                       **self.params_contour
                       )
        )
        self.dic_index_traces['contour'] = 0
        # plot frame line of latent space
        line_property = dict(color='dimgray', width=3)
        fig.add_trace(
            go.Scatter(
                x=[x_range[1], x_range[1]],
                y=y_range,
                line=line_property,
                mode='lines')
        )
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=[y_range[0], y_range[0]],
                line=line_property,
                mode='lines')
        )
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=[y_range[1], y_range[1]],
                line=line_property,
                mode='lines')
        )
        fig.add_trace(
            go.Scatter(
                x=[x_range[0], x_range[0]],
                y=y_range,
                line=line_property,
                mode='lines')
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
        self.dic_index_traces['grids'] = 5

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
        self.dic_index_traces['data'] = 6

        # draw click point initialized by visible=False
        fig.add_trace(
            go.Scatter(
                x=np.array(0.0),
                y=np.array(0.0),
                visible=False,
                marker=dict(
                    size=18,
                    symbol='x',
                    color='#e377c2',
                    line=dict(
                        width=3.0,
                        color="dimgray"
                    )
                ),
                name='clicked_point'
            )
        )
        self.dic_index_traces['clicked_point'] = 7

        fig.add_annotation(x=self.data[:, 0].max(),
                           y=self.data[:, 1].min()-0.1,
                           text=annotation_text,
                           showarrow=False,
                           font=dict(size=12),
                           xanchor='right')

        self.graph_whole = dcc.Graph(
            id=id_graph,
            figure=fig,
            config=config
        )

        self.store_fig_whole = dcc.Store(
            id=id_store,
            data=fig,
            storage_type='memory'
        )

    def set_graph_indiv(self, id_graph, id_store, config, params_figure_layout={}):
        fig = go.Figure(
            layout=go.Layout(
                yaxis={'range': [self.data.min(), self.data.max()]},
                showlegend=False,
                **params_figure_layout
            )
        )

        fig.add_trace(
            go.Bar(x=self.label_feature, y=np.zeros(self.data.shape[1]),
                   marker=dict(color='#e377c2'))
        )

        self.graph_indiv = dcc.Graph(
            id=id_graph,
            figure=fig,
            config=config
        )

        self.store_fig_indiv = dcc.Store(
            id=id_store,
            data=fig,
            storage_type='memory'
        )

    def update_trace_clicked_point(self, clickData, fig):
        if clickData is not None:
            if clickData['points'][0]['curveNumber'] == self.dic_index_traces['grids']:
                fig.update_traces(
                    x=np.array(clickData['points'][0]['x']),
                    y=np.array(clickData['points'][0]['y']),
                    visible=True,
                    marker=dict(
                        symbol='x'
                    ),
                    selector=dict(name='clicked_point', type='scatter')
                )
                # self.index_clicked_grid = clickData['points'][0]['pointIndex']
                return fig
            elif clickData['points'][0]['curveNumber'] == self.dic_index_traces['data']:
                fig.update_traces(
                    x=np.array(clickData['points'][0]['x']),
                    y=np.array(clickData['points'][0]['y']),
                    visible=True,
                    marker=dict(
                        symbol='circle-x'
                    ),
                    selector=dict(name='clicked_point', type='scatter')
                )
                return fig
                # self.index_clicked_grid = self._get_index_nearest_grid(x=clickData['points'][0]['x'],
                #                                                        y=clickData['points'][0]['y'])
            elif clickData['points'][0]['curveNumber'] == self.dic_index_traces['clicked_point']:
                fig.update_traces(
                    selector=dict(name='clicked_point', type='scatter'),
                    visible=False
                )
                return fig
                # self.index_clicked_grid = None
            else:
                return dash.no_update
        else:
            return dash.no_update

    def _get_index_nearest_grid(self, x, y):
        coordinate = np.array([x, y])[None, :]
        distance = cdist(self.grid_points, coordinate, metric='sqeuclidean')
        index_nearest = np.argmin(distance.ravel())
        return index_nearest


class LatentSpace(Space):
    pass


class ObservedSpace(Space):
    pass


class FeatureSpace(Space):
    pass
