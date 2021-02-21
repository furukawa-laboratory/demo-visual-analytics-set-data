import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from gmm_net.tools.normalize_like_map import normalize_like_map
from gmm_net.models.gmm_net_for_own_opp_team_performance import \
    GMMNetworkForOwnTeamOppTeamPerformance as GMMNet
from datasets.nba.nba import load_data
from sklearn.decomposition import PCA
import math


def calculate_bandwidth(
        n_data_in_one_sigma, n_samples, n_components, width_latent_space
):
    volume_latent_space = width_latent_space ** n_components
    if n_components == 2:
        return math.sqrt(
            (n_data_in_one_sigma * volume_latent_space) / (n_samples * math.pi)
        )
    if n_components == 4:
        return math.pow(
            (2.0 * n_data_in_one_sigma * volume_latent_space) / (n_samples * (math.pi ** 2)),
            1.0 / 4.0
        )
    else:
        raise ValueError("Not implemented n_components={}".format(n_components))


# Load dataset
version = '2'
dict_nba = load_data(version=version)
target_seasons = "2019"  # Focus seasong 2018-19
## Load dataset about members
member_features_original = dict_nba[target_seasons]["about_member"][
    "feature"
].copy()
label_member = dict_nba[target_seasons]["about_member"]["label"]["member"]
label_feature = dict_nba[target_seasons]["about_member"]["label"]["feature"]
df_member_info = dict_nba[target_seasons]["about_member"]["info"]
label_position = df_member_info['Pos'].values
## Load dataset about teams
### training data
win_team_bag_of_members_train = dict_nba[target_seasons]["about_game"]["feature"]["train"]["win"].copy()
lose_team_bag_of_members_train = dict_nba[target_seasons]["about_game"]["feature"]["train"]["lose"].copy()
win_team_performance_train = dict_nba[target_seasons]["about_game"]["target"]["train"]["win"].copy()
lose_team_performance_train = dict_nba[target_seasons]["about_game"]["target"]["train"]["lose"].copy()
n_team_train = (
        win_team_bag_of_members_train.shape[0] + lose_team_bag_of_members_train.shape[0]
)
# Calculate total playing time corresponds each member
all_bag_of_members_train = np.concatenate(
    [win_team_bag_of_members_train, lose_team_bag_of_members_train], axis=0
)
member_total_play_time = np.sum(all_bag_of_members_train, axis=0)

win_team_club_label_train = dict_nba[target_seasons]["about_game"]["label"]["team"]["train"]["win"].copy()
lose_team_club_label_train = dict_nba[target_seasons]["about_game"]["label"]["team"]["train"]["lose"].copy()
wl_team_club_label_train = np.concatenate([win_team_club_label_train, lose_team_club_label_train])
import pandas as pd

df_ranking = pd.read_csv('./datasets/nba/ranking.csv')
df_team_train = pd.DataFrame(data=wl_team_club_label_train, columns=['Short name'])
label_club_train_encoded = pd.merge(df_team_train,
                                    df_ranking,
                                    on='Short name',
                                    how='left')['W/L%'].values

# Preprocess
## for member features
prior_mean = 0.0
prior_precision = 100
after_standardization = True
member_features_map = normalize_like_map(
    X=member_features_original,
    weights=member_total_play_time,
    prior_mean=prior_mean,
    prior_precision=prior_precision,
    after_standardization=after_standardization,
)
### Negate negative stats
member_features_map[:, label_feature == "TOVPM"] *= -1
member_features_map[:, label_feature == "PFPM"] *= -1
label_feature[label_feature == "TOVPM"] = "NTOVPM"
label_feature[label_feature == "PFPM"] = "NPFPM"

## standardize performance
standardizer_to_performance = StandardScaler()
standardizer_to_performance.fit(
    np.concatenate(
        [win_team_performance_train, lose_team_performance_train], axis=0
    )
)
win_team_performance_train = standardizer_to_performance.transform(
    win_team_performance_train
)
lose_team_performance_train = standardizer_to_performance.transform(
    lose_team_performance_train
)

# set common parameter
seed = 13
is_save_history = False
width_latent_space = 2.0
path_joblib = "./dumped/"

## about lower ukr
params_lower_ukr = {}
params_lower_ukr["n_components"] = 2
n_data_in_one_sigma_ukr = 4
params_lower_ukr["bandwidth_gaussian_kernel"] = calculate_bandwidth(
    n_data_in_one_sigma=n_data_in_one_sigma_ukr,
    n_samples=member_features_map.shape[0],
    n_components=params_lower_ukr["n_components"],
    width_latent_space=width_latent_space,
)
params_lower_ukr["is_compact"] = True
params_lower_ukr["lambda_"] = 0.0
pca = PCA(n_components=params_lower_ukr["n_components"],
          random_state=seed)
params_lower_ukr["init"] = pca.fit_transform(member_features_map)
params_lower_ukr["init"] *= params_lower_ukr["bandwidth_gaussian_kernel"] * 0.05

params_lower_ukr["is_loocv"] = False
params_lower_ukr["is_save_history"] = is_save_history
params_lower_ukr["weights"] = member_total_play_time
nb_epoch_ukr = 30000
eta_ukr = 0.03

## about ukr_for_kde
params_upper_ukr_kde = {}
params_upper_ukr_kde["n_embedding"] = 2
params_upper_ukr_kde["bandwidth_kde"] = 0.2
n_data_in_one_sigma_ukr_kde = 4
params_upper_ukr_kde["bandwidth_nadaraya"] = calculate_bandwidth(
    n_data_in_one_sigma=n_data_in_one_sigma_ukr_kde,
    n_samples=n_team_train,
    n_components=params_upper_ukr_kde["n_embedding"],
    width_latent_space=width_latent_space,
)
params_upper_ukr_kde["lambda_"] = 0.0
params_upper_ukr_kde["metric_evaluation_method"] = "quadrature_by_parts"
params_upper_ukr_kde["metric"] = 'kl'
params_upper_ukr_kde["resolution_quadrature"] = 30
params_upper_ukr_kde["init"] = "pca_densities"
params_upper_ukr_kde["random_state"] = seed

## about gplvm
params_gplvm = {}
length = calculate_bandwidth(n_data_in_one_sigma=3,
                             n_samples=n_team_train,
                             n_components=params_upper_ukr_kde["n_embedding"] * 2,
                             width_latent_space=width_latent_space)
params_gplvm["sqlength"] = length ** 2.0
params_gplvm["beta_inv"] = 0.72
params_gplvm["is_optimize_sqlength"] = False
params_gplvm["is_optimize_beta_inv"] = False
params_gplvm["how_calculate_inv"] = 'cholesky'

## about multiview
nb_epoch_multiview = 8000
learning_rate_multiview = 0.001
ratio_ukr_for_kde = 0.65
is_compact = True

## about gaussian process regression
params_gpr = {}
params_gpr["alpha"] = 0.0
params_gpr["random_state"] = seed
params_gpr["normalize_y"] = False
params_gpr["optimizer"] = None  # Not optimize

if not os.path.exists(path_joblib):
    os.mkdir(path_joblib)
path_whole_model_joblib = (
        path_joblib
        + "fit_gmm_net_seed{}_epoch{}.cmp".format(seed,
                                                  nb_epoch_multiview)
)
if os.path.exists(path_whole_model_joblib):
    print("saved model exists")
    f = open(path_whole_model_joblib, "rb")
    gmm_net = joblib.load(f)
    f.close()
else:
    print("saved model does not exist")
    # create instance and fit
    gmm_net = GMMNet(
        win_team_bag_of_members=win_team_bag_of_members_train,
        lose_team_bag_of_members=lose_team_bag_of_members_train,
        win_team_performance=win_team_performance_train,
        lose_team_performance=lose_team_performance_train,
        member_features=member_features_map,
        params_lower_ukr=params_lower_ukr,
        params_upper_ukr_kde=params_upper_ukr_kde,
        params_gplvm=params_gplvm,
        params_gpr=params_gpr,
        is_save_history=is_save_history,
        is_compact=is_compact
    )
    gmm_net.fit(
        nb_epoch_lower_ukr=nb_epoch_ukr,
        eta_lower_ukr=eta_ukr,
        nb_epoch_multiview_mm=nb_epoch_multiview,
        eta_multiview_mm=learning_rate_multiview,
        ratio_ukr_for_kde=ratio_ukr_for_kde,
        verbose=True
    )
    # save whole model
    print("whole model has been saved as pickle")
    f = open(path_whole_model_joblib, "wb")
    joblib.dump(gmm_net, f, compress=True)
    f.close()

# set parameters to visualize
resolution = 30
cmap_density = "Greys"
cmap_feature = "RdBu_r"
cmap_ccp = "RdBu_r"
is_member_cp_middle_color_zero = True
is_ccp_middle_color_zero = True
dict_position_marker = {
    'C': 'circle',
    'PF': 'square',
    'SG': 'pentagon',
    'PG': 'triangle-up',
    'SF': 'star'
}
import matplotlib.pyplot as plt
import matplotlib
cmap_member_scat = plt.get_cmap('tab10')
dict_position_color = {
    'C': cmap_member_scat(0),
    'PF': cmap_member_scat(4),
    'SG': cmap_member_scat(3),
    'PG': cmap_member_scat(2),
    'SF': cmap_member_scat(1)
}
for key in dict_position_color.keys():
    dict_position_color[key] = matplotlib.colors.to_hex(dict_position_color[key])

dict_position_correct_name = {
    'C': 'Center',
    'PF': 'Power forward',
    'SG': 'Shooting guard',
    'PG': 'Point guard',
    'SF': 'Small forward'
}
position_symbol = []
position_color = []
for lp in label_position:
    position_symbol.append(dict_position_marker[lp])
    position_color.append(dict_position_color[lp])

# position_symbol = np.array(position_symbol)
# position_color = np.array(position_color)
# params_init_lower_ukr = {
#     'marker': marker_position,
#     'dict_marker_label': dict_marker_label,
#     'params_scatter': {'alpha': 0.7,
#                        's': 17}
# }
params_init_lower_ukr=dict(
    params_scat_z=dict(
        name='member',
        marker=dict(
            size=13,
            color=position_color,
            line=dict(
                width=1.5,
                color="white"
            ),
            symbol=position_symbol
        )
    )
)
params_init_upper_ukr=dict(
    params_scat_z=dict(
        name='team',
        marker=dict(
            size=8,
            opacity=0.7,
            colorscale='Turbo_r',
            color=label_club_train_encoded
        )
    )
)
# params_init_upper_ukr = {
# 'params_scatter_latent_space': {'c': label_club_train_encoded,
#                                 'cmap': 'rainbow_r',
#                                     's': 5,
#                                     'alpha': 0.5}
# }
n_epoch_to_change_member = 700
learning_rate_to_change_member = 0.002
path_meshed = path_joblib + 'meshes_to_ccp_resolution{}.npz'.format(resolution)
if os.path.exists(path_meshed):
    npz_meshes = np.load(path_meshed)
    gmm_net.mesh_grid_mapping = npz_meshes['mapping']
    gmm_net.mesh_grid_precision = npz_meshes['precision']


app = gmm_net.define_dash_app(
    n_grid_points=resolution,
    cmap_density=cmap_density,
    cmap_feature=cmap_feature,
    cmap_ccp=cmap_ccp,
    label_member=label_member,
    label_feature=label_feature,
    label_team=wl_team_club_label_train,
    label_performance=dict_nba[target_seasons]["about_game"]["label"]["target"],
    is_member_cp_middle_color_zero=is_member_cp_middle_color_zero,
    is_ccp_middle_color_zero=is_ccp_middle_color_zero,
    params_init_lower_ukr=params_init_lower_ukr,
    params_init_upper_ukr=params_init_upper_ukr,
    n_epoch_to_change_member=n_epoch_to_change_member,
    learning_rate_to_change_member=learning_rate_to_change_member
)
server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)
    # gmm_net.visualize(
    #     n_grid_points=resolution,
    #     cmap_density=cmap_density,
    #     cmap_feature=cmap_feature,
    #     cmap_ccp=cmap_ccp,
    #     label_member=label_member,
    #     label_feature=label_feature,
    #     label_team=wl_team_club_label_train,
    #     label_performance=dict_nba[target_seasons]["about_game"]["label"]["target"],
    #     is_member_cp_middle_color_zero=is_member_cp_middle_color_zero,
    #     is_ccp_middle_color_zero=is_ccp_middle_color_zero,
    #     params_init_lower_ukr=params_init_lower_ukr,
    #     params_init_upper_ukr=params_init_upper_ukr,
    #     n_epoch_to_change_member=n_epoch_to_change_member,
    #     learning_rate_to_change_member=learning_rate_to_change_member
    # )

    if not os.path.exists(path_meshed):
        np.savez_compressed(path_meshed,
                            mapping=gmm_net.mesh_grid_mapping,
                            precision=gmm_net.mesh_grid_precision)
