from gmm_net.models.core import ObservedSpace
import numpy as np
class DoubleDomainGMM(object):
    def __init__(self, data):
        self.mesh_grid_mapping = None
        self.mesh_grid_precision = None
        self.own_ls = None
        self.opp_ls = None
        self.os = ObservedSpace(data=data)

    def define_graphs(self,own_ls, opp_ls, label_feature):
        self.own_ls = own_ls
        self.opp_ls = opp_ls
        self.os.label_feature = label_feature
