import pytest
from lstchain.io import DataSelection, DataBinning


def test_data_selection(simulated_dl2_file):
    from lstchain.io import read_mc_dl2_to_QTable

    tempsel = DataSelection()
    tempsel.event_filters = {
        "intensity": [0, 1000],
        "width": [0, 100],
        "length": [0, 100],
        "r": [0, 1],
        "wl": [0.1, 1],
        "leakage_intensity_width_2": [0, 1]
    }
    tempsel.fixed_gh_cut = 0.5
    tempsel.fixed_theta_cut = 1
    tempsel.irf_obs_time = 50
    tempsel.allowed_tels = [1]

    data, _ = read_mc_dl2_to_QTable(simulated_dl2_file)

    data_filter = tempsel.filter_cut(data)
    data_gh = tempsel.gh_cut(data)
    data_tel = tempsel.allowed_tels_filter(data)

    assert data_filter["intensity"].max() < 1000
    assert data_gh["gh_score"].max() > 0.5
    assert data_tel["tel_id"].mean() == 1


def test_data_binning():
    tempbin = DataBinning()

    tempbin.true_energy_min = 0.01
    tempbin.true_energy_max = 100
    tempbin.true_energy_n_bins_per_decade = 5.5
    tempbin.reco_energy_min = 0.01
    tempbin.reco_energy_max = 100
    tempbin.reco_energy_n_bins_per_decade = 5.5
    tempbin.energy_migration_min = 0.2
    tempbin.energy_migration_max = 5
    tempbin.energy_migration_n_bins = 15
    tempbin.fov_offset_min = 0.1
    tempbin.fov_offset_max = 1.1
    tempbin.fov_offset_n_edges = 9
    tempbin.bkg_fov_offset_min = 0
    tempbin.bkg_fov_offset_max = 10
    tempbin.bkg_fov_offset_n_edges = 11
    tempbin.source_offset_min = 0
    tempbin.source_offset_max = 1.0001
    tempbin.source_offset_n_edges = 1001

    e_true = tempbin.true_energy_bins()
    e_reco = tempbin.reco_energy_bins()
    e_migra = tempbin.energy_migration_bins()
    fov_off = tempbin.fov_offset_bins()
    bkg_fov = tempbin.bkg_fov_offset_bins()
    src_off = tempbin.source_offset_bins()

    assert len(e_true) == 22
    assert len(e_reco) == 22
    assert len(e_migra) == 15
    assert len(fov_off) == 9
    assert len(bkg_fov) == 11
    assert len(src_off) == 1001
