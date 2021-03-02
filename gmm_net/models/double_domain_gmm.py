from gmm_net.models.core import ObservedSpace
import numpy as np
import dash
from scipy.spatial.distance import cdist


class DoubleDomainGMM(object):
    def __init__(self, data):
        self.mesh_grid_mapping = None
        self.mesh_grid_precision = None
        self.os = ObservedSpace(data=data)
        self.dic_ls = {}
        self.params_contour = {}

    def define_graphs(self, own_ls, opp_ls, label_feature, id_fb,
                      params_contour,
                      params_figure_layout={}):
        import plotly.graph_objects as go
        import dash_core_components as dcc

        config = {'displayModeBar': False}
        self.dic_ls = {'own': own_ls, 'opp': opp_ls}
        self.os.label_feature = label_feature
        self.os.set_graph_indiv(id_graph=id_fb,
                                id_store=id_fb+'_fig_store',
                                params_figure_layout=params_figure_layout,
                                config=config)
        for key, ls in self.dic_ls.items():
            ls.dropdown = dcc.Dropdown(
                id=key+'_team_dropdown',
                options=[{"value": i, "label": x}
                         for i, x in enumerate(self.os.label_feature)],
                placeholder="Select own team performance shown as contour",
                style={'width': '95%', 'margin': '0 auto'},
                clearable=True
            )
        self.params_contour = params_contour

    def update_bar(self, own_clickData, opp_clickData):
        import dash
        ctx = dash.callback_context
        if not ctx.triggered or ctx.triggered[0]['value'] is None:
            return dash.no_update
        else:
            # callback of this method is assumpted that update ls method is called before this method is called
            print('in update_bar')
            print("own_clickData={}".format(own_clickData))
            print("opp_clickData={}".format(opp_clickData))
            if self.dic_ls['own'].index_clicked_grid is not None:
                # own team map is clicked
                if self.dic_ls['opp'].index_clicked_grid is not None:
                    # opp team map is clicked
                    self.os.graph_indiv.figure.update_traces(
                        y=self.mesh_grid_mapping[self.dic_ls['own'].index_clicked_grid, self.dic_ls['opp'].index_clicked_grid, :]
                    )
                else:
                    # opp team map is not clicked
                    self.os.graph_indiv.figure.update_traces(
                        y=np.mean(self.mesh_grid_mapping[self.dic_ls['own'].index_clicked_grid, : , :], axis=0)
                    )
            else:
                if self.dic_ls['opp'].index_clicked_grid is not None:
                    self.os.graph_indiv.figure.update_traces(
                        y=np.mean(self.mesh_grid_mapping[:, self.dic_ls['opp'].index_clicked_grid, :], axis=0)
                    )
                else:
                    self.os.graph_indiv.figure.update_traces(
                        y=np.zeros(self.mesh_grid_mapping.shape[2])
                    )

            return self.os.graph_indiv.figure


    def update_ls(self, index_selected_feature, clickData, which_update: str):
        import dash
        ctx = dash.callback_context
        if not ctx.triggered or ctx.triggered[0]['value'] is None:
            return dash.no_update
        else:
            if which_update == 'own':
                ls_updated = self.dic_ls['own']
                ls_triggered = self.dic_ls['opp']
                # which_triggerd = 'opp'
            elif which_update == 'opp':
                ls_updated = self.dic_ls['opp']
                ls_triggered = self.dic_ls['own']
                # which_triggerd = 'own'
            else:
                raise ValueError('invalid which_update={}'.format(which_update))
            print('in own_opp_gplvm.update_ls')
            print('index_selected_feature={}'.format(index_selected_feature))
            print('clickData={}'.format(clickData))
            print('which={}'.format(which_update))

            if index_selected_feature is not None:
                ls_triggered.update_trace_clicked_point(clickData=clickData)
                if ls_triggered.index_clicked_grid is not None:
                    # set value to conditional component plane
                    index_nearest_grid = ls_triggered.index_clicked_grid
                    if which_update == 'own':
                        grid_value = self.mesh_grid_mapping[:, ls_triggered.index_clicked_grid, index_selected_feature]
                    else:
                        grid_value = self.mesh_grid_mapping[index_nearest_grid, :, index_selected_feature]
                else:
                    # set value to marginal
                    if which_update == 'own':
                        grid_value = np.mean(self.mesh_grid_mapping[:, :, index_selected_feature], axis=1)
                    else:
                        grid_value = np.mean(self.mesh_grid_mapping[:, :, index_selected_feature], axis=0)
                ls_updated.graph_whole.figure.update_traces(
                    selector=dict(type='contour'),
                    z=grid_value,
                    **self.params_contour
                )
                return ls_updated.graph_whole.figure
            else:
                ls_updated.graph_whole.figure.update_traces(
                    z=None,
                    selector=dict(type='contour')
                )
                return ls_updated.graph_whole.figure
    def _get_index_nearest_grid(self, x, y, which: str):
        coordinate = np.array([x, y])[None, :]
        grids = self.dic_ls[which].grid_points

        distance = cdist(grids, coordinate, metric='sqeuclidean')
        index_nearest = np.argmin(distance.ravel())
        return index_nearest



