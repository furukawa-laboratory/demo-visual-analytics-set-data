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
        self.os.set_graph_indiv(id=id_fb,
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
        print('in update_bar')
        print("own_clickData={}".format(own_clickData))
        print("opp_clickData={}".format(opp_clickData))
        if own_clickData is not None and opp_clickData is not None:
            index_own_clickdata = own_clickData['points'][0]['pointIndex']
            index_opp_clickdata = opp_clickData['points'][0]['pointIndex']

            # check which trace is clicked in own team map
            if own_clickData['points'][0]['curveNumber'] == self.dic_ls['own'].dic_index_traces['grids']:
                index_own_nearest_grid = index_own_clickdata
            elif own_clickData['points'][0]['curveNumber'] == self.dic_ls['own'].dic_index_traces['data']:
                index_own_nearest_grid = self._get_index_nearest_grid(
                    x=own_clickData['points'][0]['x'],
                    y=own_clickData['points'][0]['y'],
                    which='own'
                )
                print(
                    'own nearest grid={}th grid {}'.format(
                        index_own_nearest_grid,
                        self.dic_ls['own'].grid_points[index_own_nearest_grid]
                    )
                )
            else:
                return dash.no_update
            # check which trace is clicked in opp team map
            if opp_clickData['points'][0]['curveNumber'] == self.dic_ls['opp'].dic_index_traces['grids']:
                index_opp_nearest_grid = index_opp_clickdata
            elif opp_clickData['points'][0]['curveNumber'] == self.dic_ls['opp'].dic_index_traces['data']:
                index_opp_nearest_grid = self._get_index_nearest_grid(
                    x=opp_clickData['points'][0]['x'],
                    y=opp_clickData['points'][0]['y'],
                    which='opp'
                )
            else:
                return dash.no_update
            self.os.graph_indiv.figure.update_traces(
                y=self.mesh_grid_mapping[index_own_nearest_grid, index_opp_nearest_grid, :]
            )
            return self.os.graph_indiv.figure

        else:
            return dash.no_update

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
                if clickData is not None:
                    # update clicked point in map triggered
                    ls_triggered.update_trace_clicked_point(clickData=clickData)
                    # print('index={}'.format(index))
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
                    return dash.no_update
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



