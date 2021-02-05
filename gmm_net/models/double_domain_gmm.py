from gmm_net.models.core import ObservedSpace
import numpy as np

class DoubleDomainGMM(object):
    def __init__(self, data):
        self.mesh_grid_mapping = None
        self.mesh_grid_precision = None
        self.os = ObservedSpace(data=data)
        self.dic_ls = {}

    def define_graphs(self, own_ls, opp_ls, label_feature, id_fb,
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
                clearable=True
            )
        pass

    def update_ls(self, index_selected_feature, clickData, which_update: str):
        import dash
        if which_update == 'own':
            ls_updated = self.dic_ls['own']
            ls_triggered = self.dic_ls['opp']
        elif which_update == 'opp':
            ls_updated = self.dic_ls['opp']
            ls_triggered = self.dic_ls['own']
        else:
            raise ValueError('invalid which_update={}'.format(which_update))
        print('in own_opp_gplvm.update_ls')
        print('index_selected_feature={}'.format(index_selected_feature))
        print('clickData={}'.format(clickData))
        print('which={}'.format(which_update))

        if index_selected_feature is not None:
            if clickData is not None:
                index_clickdata = clickData['points'][0]['pointIndex']
                # print('index={}'.format(index))
                if clickData['points'][0]['curveNumber'] == ls_triggered.dic_index_traces['data']:
                    # print('clicked latent variable')
                    # if latent variable is clicked
                    # self.os.graph_indiv.figure.update_traces(y=self.X[index])
                    grid_value = None
                elif clickData['points'][0]['curveNumber'] == ls_triggered.dic_index_traces['grids']:
                    # self.os.graph_indiv.figure.update_traces(y=self.ls.grid_mapping[index])
                    if which_update == 'own':
                        grid_value = self.mesh_grid_mapping[:, index_clickdata, index_selected_feature]
                    else:
                        grid_value = self.mesh_grid_mapping[index_clickdata, :, index_selected_feature]

                ls_updated.graph_whole.figure.update_traces(
                    z=grid_value,
                    selector=dict(type='contour', name='contour')
                )

                return ls_updated.graph_whole.figure
            else:
                return dash.no_update
        else:
            ls_updated.graph_whole.figure.update_traces(
                z=None,
                selector=dict(type='contour', name='contour')
            )
            return ls_updated.graph_whole.figure


