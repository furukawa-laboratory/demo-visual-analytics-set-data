from gmm_net.models.core import ObservedSpace
import numpy as np
import dash
from scipy.spatial.distance import cdist
import plotly.graph_objects as go


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
                                id_store=id_fb + '_fig_store',
                                params_figure_layout=params_figure_layout,
                                config=config)
        for key, ls in self.dic_ls.items():
            ls.dropdown = dcc.Dropdown(
                id=key + '_team_dropdown',
                options=[{"value": i, "label": x}
                         for i, x in enumerate(self.os.label_feature)],
                placeholder="Select own team performance shown as contour",
                style={'width': '95%', 'margin': '0 auto'},
                clearable=True
            )
        self.params_contour = params_contour

    def update_bar(self, fig_own_ls, fig_opp_ls, prev_fig_bar_json):
        import dash
        ctx = dash.callback_context
        if not ctx.triggered or ctx.triggered[0]['value'] is None:
            return dash.no_update
        else:
            # callback of this method is assumpted that update ls method is called before this method is called
            # print('in update_bar')
            # print("own_clickData={}".format(own_clickData))
            # print("opp_clickData={}".format(opp_clickData))
            fig_bar = go.Figure(**prev_fig_bar_json)
            own_index_trace_clicked_point = self.dic_ls['own'].dic_index_traces['clicked_point']
            opp_index_trace_clicked_point = self.dic_ls['opp'].dic_index_traces['clicked_point']
            if fig_own_ls.data[own_index_trace_clicked_point].visible:
                # own team map is clicked
                own_index_nearest_grid = self.dic_ls['own']._get_index_nearest_grid(
                    x=fig_own_ls.data[own_index_trace_clicked_point].x[0],
                    y=fig_own_ls.data[own_index_trace_clicked_point].y[0]
                )
                if fig_opp_ls.data[opp_index_trace_clicked_point].visible:
                    # opp team map is clicked
                    opp_index_nearest_grid = self.dic_ls['opp']._get_index_nearest_grid(
                        x=fig_opp_ls.data[opp_index_trace_clicked_point].x[0],
                        y=fig_opp_ls.data[opp_index_trace_clicked_point].y[0]
                    )
                    fig_bar.update_traces(
                        y=self.mesh_grid_mapping[
                          own_index_nearest_grid,
                          opp_index_nearest_grid,
                          :
                          ]
                    )
                else:
                    fig_bar.update_traces(
                        y=np.mean(
                            self.mesh_grid_mapping[own_index_nearest_grid, :, :],
                            axis=0
                        )
                    )
            else:
                if fig_opp_ls.data[opp_index_trace_clicked_point].visible:
                    # opp team map is clicked
                    opp_index_nearest_grid = self.dic_ls['opp']._get_index_nearest_grid(
                        x=fig_opp_ls.data[opp_index_trace_clicked_point].x[0],
                        y=fig_opp_ls.data[opp_index_trace_clicked_point].y[0]
                    )
                    fig_bar.update_traces(
                        y=np.mean(
                            self.mesh_grid_mapping[:, opp_index_nearest_grid, :],
                            axis=0
                        )
                    )
                else:
                    fig_bar.update_traces(
                        y=np.zeros(self.mesh_grid_mapping.shape[2])
                    )
            return fig_bar

    def update_ls(self, index_selected_feature,
                  fig_own_ls, fig_opp_ls,
                  which_update: str):
        ctx = dash.callback_context
        if not ctx.triggered or ctx.triggered[0]['value'] is None:
            return dash.no_update
        else:
            if which_update == 'own':
                fig_ls_updated = fig_own_ls
                fig_ls_triggered = fig_opp_ls
                which_triggered = 'opp'
            elif which_update == 'opp':
                fig_ls_updated = fig_opp_ls
                fig_ls_triggered = fig_own_ls
                which_triggered = 'own'
            else:
                raise ValueError('invalid which_update={}'.format(which_update))
            print('in own_opp_gplvm.update_ls')
            print('index_selected_feature={}'.format(index_selected_feature))
            print('which={}'.format(which_update))

            if index_selected_feature is not None:
                index_trace_clicked_point = self.dic_ls[which_triggered].dic_index_traces['clicked_point']
                if fig_ls_triggered.data[index_trace_clicked_point].visible:
                    # if clicked point exists, set value to conditional component plane
                    index_nearest_grid = self.dic_ls[which_triggered]._get_index_nearest_grid(
                        x=fig_ls_triggered.data[index_trace_clicked_point].x[0],
                        y=fig_ls_triggered.data[index_trace_clicked_point].y[0]
                    )
                    if which_update == 'own':
                        grid_value = self.mesh_grid_mapping[
                                     :,
                                     index_nearest_grid,
                                     index_selected_feature
                                     ]
                    else:
                        grid_value = self.mesh_grid_mapping[
                                     index_nearest_grid,
                                     :,
                                     index_selected_feature
                                     ]
                else:
                    # If clicked point does not exist, set value to marginal component plance
                    if which_update == 'own':
                        grid_value = np.mean(
                            self.mesh_grid_mapping[:, :, index_selected_feature],
                            axis=1
                        )
                    else:
                        grid_value = np.mean(
                            self.mesh_grid_mapping[:, :, index_selected_feature],
                            axis=0
                        )
                # update
                fig_ls_updated.update_traces(
                    selector=dict(type='contour'),
                    z=grid_value,
                    **self.params_contour
                )
                return fig_ls_updated

            else:
                fig_ls_updated.update_traces(
                    z=None,
                    selector=dict(type='contour')
                )
                return fig_ls_updated

    def _get_index_nearest_grid(self, x, y, which: str):
        coordinate = np.array([x, y])[None, :]
        grids = self.dic_ls[which].grid_points

        distance = cdist(grids, coordinate, metric='sqeuclidean')
        index_nearest = np.argmin(distance.ravel())
        return index_nearest
