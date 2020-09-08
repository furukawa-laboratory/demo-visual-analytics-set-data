import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from gmm_net.tools.normalize_like_map import normalize_like_map
from gmm_net.models.own_opp_team_performance_multilevel_view_mm import \
    OwnTeamOppTeamPerformanceMultiLevelViewMM as TTPMVMM
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


def _main():
    # load dataset
    version = '2'
    dict_nba = load_data(version=version)
    target_seasons = "2019"  # 現状は2018-2019だけにフォーカス
    ## 各メンバーについてのデータを引き出す
    member_features_original = dict_nba[target_seasons]["about_member"][
        "feature"
    ].copy()
    label_member = dict_nba[target_seasons]["about_member"]["label"]["member"]
    label_feature = dict_nba[target_seasons]["about_member"]["label"]["feature"]
    df_member_info = dict_nba[target_seasons]["about_member"]["info"]
    label_position = df_member_info['Pos'].values
    ## 次にチーム関連のデータ
    ### training data
    win_team_bag_of_members_train = dict_nba[target_seasons]["about_game"]["feature"][
        "train"
    ]["win"].copy()
    lose_team_bag_of_members_train = dict_nba[target_seasons]["about_game"]["feature"][
        "train"
    ]["lose"].copy()
    win_team_performance_train = dict_nba[target_seasons]["about_game"]["target"][
        "train"
    ]["win"].copy()
    lose_team_performance_train = dict_nba[target_seasons]["about_game"]["target"][
        "train"
    ]["lose"].copy()
    n_team_train = (
            win_team_bag_of_members_train.shape[0] + lose_team_bag_of_members_train.shape[0]
    )
    # training dataにおける各メンバーの総出場時間を計算しておく
    all_bag_of_members_train = np.concatenate(
        [win_team_bag_of_members_train, lose_team_bag_of_members_train], axis=0
    )
    member_total_play_time = np.sum(all_bag_of_members_train, axis=0)

    win_team_club_label_train = dict_nba[target_seasons]["about_game"]["label"]["team"]["train"]["win"].copy()
    lose_team_club_label_train = dict_nba[target_seasons]["about_game"]["label"]["team"]["train"]["lose"].copy()
    wl_team_club_label_train = np.concatenate([win_team_club_label_train, lose_team_club_label_train])
    import pandas as pd
    df_ranking = pd.read_csv('./ranking.csv')
    df_team_train = pd.DataFrame(data=wl_team_club_label_train, columns=['Short name'])
    label_club_train_encoded = pd.merge(df_team_train,
                                        df_ranking,
                                        on='Short name',
                                        how='left')['W/L%'].values
    # le = LabelEncoder()
    # label_club_train_encoded = le.fit_transform(wl_team_club_label_train)

    win_team_club_label_test = dict_nba[target_seasons]["about_game"]["label"]["team"]["test"]["win"].copy()
    lose_team_club_label_test = dict_nba[target_seasons]["about_game"]["label"]["team"]["test"]["lose"].copy()
    wl_team_club_label_test = np.concatenate([win_team_club_label_test, lose_team_club_label_test])
    df_team_test = pd.DataFrame(data=wl_team_club_label_test, columns=['Short name'])
    label_club_test_encoded = pd.merge(df_team_test,
                                       df_ranking,
                                       on='Short name',
                                       how='left')['W/L%'].values

    ## 次にtest data
    win_team_bag_of_members_test = dict_nba[target_seasons]["about_game"]["feature"]["test"]["win"].copy()
    lose_team_bag_of_members_test = dict_nba[target_seasons]["about_game"]["feature"]["test"]["lose"].copy()
    win_team_performance_test = dict_nba[target_seasons]["about_game"]["target"]["test"]["win"].copy()
    lose_team_performance_test = dict_nba[target_seasons]["about_game"]["target"]["test"]["lose"].copy()
    n_game_test = win_team_bag_of_members_test.shape[0]
    ### テストで与えるために整形する
    own_bag_of_members_test = np.concatenate([win_team_bag_of_members_test,
                                              lose_team_bag_of_members_test], axis=0)
    team_performance_test = np.concatenate([win_team_performance_test,
                                            lose_team_performance_test], axis=0)
    # 前処理
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
    ### ネガティブなスタッツの符号を変換
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
    team_performance_test = standardizer_to_performance.transform(team_performance_test)

    # set common parameter
    seed = 13
    is_save_history = False
    is_save_whole_pkl = True
    width_latent_space = 2.0  # こういう仕様に下位のUKRも上位のUKRもなっているのでここで決めとく
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
    params_lower_ukr["weights"] = member_total_play_time  # UKRの重みを選手の総出場時間に
    nb_epoch_ukr = 30000
    eta_ukr = 0.03

    ## about ukr_for_kde
    params_upper_ukr_kde = {}
    params_upper_ukr_kde["n_embedding"] = 2
    # list_bandwidth_kde = [0.2]
    params_upper_ukr_kde["bandwidth_kde"] = 0.2
    n_data_in_one_sigma_ukr_kde = 4
    params_upper_ukr_kde["bandwidth_nadaraya"] = calculate_bandwidth(
        n_data_in_one_sigma=n_data_in_one_sigma_ukr_kde,
        n_samples=n_team_train,
        n_components=params_upper_ukr_kde["n_embedding"],
        width_latent_space=width_latent_space,
    )
    # params_upper_ukr_kde["is_compact"] = True
    params_upper_ukr_kde["lambda_"] = 0.0
    params_upper_ukr_kde["metric_evaluation_method"] = "quadrature_by_parts"
    params_upper_ukr_kde["metric"] = 'kl'
    params_upper_ukr_kde["resolution_quadrature"] = 30
    params_upper_ukr_kde["init"] = "pca_densities"
    # params_upper_ukr_kde["is_save_history"] = is_save_history
    params_upper_ukr_kde["random_state"] = seed
    # init_upper_ukr_kde = None

    ## about gplvm
    params_gplvm = {}
    length = calculate_bandwidth(n_data_in_one_sigma=3,
                                 n_samples=n_team_train,
                                 n_components=params_upper_ukr_kde["n_embedding"]*2,
                                 width_latent_space=width_latent_space)
    params_gplvm["sqlength"] = length ** 2.0
    params_gplvm["beta_inv"] = 0.72 # この辺のハイパーパラメータは何がいいのか良くわからんのでとりあえずこれで
    params_gplvm["is_optimize_sqlength"] = False
    params_gplvm["is_optimize_beta_inv"] = False
    params_gplvm["how_calculate_inv"] = 'cholesky'
    # params_gplvm["is_compact"] = True

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


    # create instance and fit
    if not os.path.exists(path_joblib):
        os.mkdir(path_joblib)
    path_whole_model_joblib = (
            path_joblib
            + "fit_team_team_performance_mm_seed{}_epoch{}.cmp".format(seed,
                                                                       nb_epoch_multiview)
    )
    # 学習済みのモデルがpklとして保存されている場合はそれを用いる
    if os.path.exists(path_whole_model_joblib):
        print("whole pickle exists")
        f = open(path_whole_model_joblib, "rb")
        team_team_performance_mm = joblib.load(f)
        f.close()
    # 保存されていない場合は新たにインスタンスを作成し学習する
    else:
        print("whole pickle does not exist")
        # fit whole model
        team_team_performance_mm = TTPMVMM(
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
        team_team_performance_mm.fit(
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
        joblib.dump(team_team_performance_mm, f)
        f.close()

    # f = open(path_whole_model_joblib, "rb")
    # team_team_performance_mm = pickle.load(f)
    # f.close()
    #
    # del team_team_performance_mm.history
    # del team_team_performance_mm.lower_ukr.history
    # del team_team_performance_mm.upper_ukr_kde.history
    # path_whole_model_pkl = (
    #         path_pkl
    #         + "fit_team_team_performance_mm_seed{}_ratio_ukr{}_epoch{}_nohistory.pkl".format(seed,
    #                                                                                          text_ratio,
    #                                                                                          nb_epoch_multiview)
    # )
    # f = open(path_whole_model_pkl, "wb")
    # pickle.dump(team_team_performance_mm, f)
    # f.close()
    # save delete history
    # path_whole_model_joblib = (
    #         path_joblib
    #         + "fit_team_team_performance_mm_seed{}_ratio_ukr{}_epoch{}_nohistory.comp".format(seed,
    #                                                                                          text_ratio,
    #                                                                                          nb_epoch_multiview)
    # )
    # import joblib
    # f = open(path_whole_model_joblib, "wb")
    # joblib.dump(team_team_performance_mm, f, compress=True)
    # f.close()
    #
    # f = open(path_whole_model_joblib, "rb")
    # team_team_performance_mm = joblib.load(f)
    # f.close()

    # visualize interactively
    n_grid_points = 15
    cmap_density = "binary"
    cmap_feature = "bwr"
    cmap_ccp = "bwr"
    is_member_cp_middle_color_zero = True
    is_ccp_middle_color_zero = True
    dict_position_marker = {
        'C': 'o',
        'PF': ',',
        'SG': 'p',
        'PG': '^',
        'SF': '*'
    }
    marker_position = []
    for lp in label_position:
        marker_position.append(dict_position_marker[lp])
    params_init_lower_ukr = {
        'marker': marker_position,
        'params_scatter': {'alpha': 0.7,
                           's': 17}
    }
    params_init_upper_ukr = {
        'params_scatter_latent_space': {'c': label_club_train_encoded,
                                        'cmap': 'jet',
                                        's': 5,
                                        'alpha': 0.5}
    }
    n_epoch_to_change_member = 700
    learning_rate_to_change_member = 0.002
    path_meshed = path_joblib + 'meshes_to_ccp_resolution{}.npz'.format(n_grid_points)
    if os.path.exists(path_meshed):
        npz_meshes = np.load(path_meshed)
        team_team_performance_mm.mesh_grid_mapping = npz_meshes['mapping']
        team_team_performance_mm.mesh_grid_precision = npz_meshes['precision']

    team_team_performance_mm.visualize(
        n_grid_points=n_grid_points,
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
    np.savez_compressed(path_meshed,
                        mapping=team_team_performance_mm.mesh_grid_mapping,
                        precision=team_team_performance_mm.mesh_grid_precision)


    print("finish!")


if __name__ == "__main__":
    _main()
