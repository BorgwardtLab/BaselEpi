#!/uisr/bin/env python3
#
# Creates graphs based on the traffic model of the catonal traffic
# department of Basel Stadt. This includes
#
# (i)   reading the transport matrices (foot, bike, public, car)
# (ii)  loading spatial partitions of Basel Stadt (quarters, socioeconomic, ...)
# (iii) associating the cells of the transport model with the partition areas
# (iv)  inferring in- and between-area mobility
# (v)   constructing mobility graphs
#
# Further extracts traffic timeseries from https://data.bs.ch.
#
### PACKAGES ##################################################################

import geopandas         as     gpd
import igraph            as     ig
import json
import matplotlib        as     mpl
import matplotlib.image  as     mim
import matplotlib.pyplot as     plt
import numpy             as     np
import os
import pandas            as     pd
import requests
from datetime import datetime, timedelta
from pyproj import CRS
from shapely.geometry import Point
from tqdm import tqdm

### GLOBAL VARIABLES ##########################################################

main_dir = os.getcwd()
sbbbvb_dir = 'SBB Data'
tmsrs_dir = '../../../data/shapefiles/'
shape_dir = '../../../data/shapefiles/GeodatenBS_20200605/Wohnviertel/'
trans_dir = '../../../data/BS_Vehrkersdepartement_Redle/2020_08_12_Aggregierte_Matrizen_GVM_Region_Basel/Gebietseinteilung_frei/Matrizen_CbyC/'
zone_dir = '../../../data/BS_Vehrkersdepartement_Redle/2020_07_17_Daten_GVM_Region_Basel/'
graph_dir = 'graphs/'
figure_dir = '../../../figures/'
output_dir = 'output'


### MAIN ######################################################################

def main():
    originals = True
    bootstrap = False
    n_boots = 33
    n = 3

    # ts_df     = compile_total_ts(criterion,mode,n=3)

    if originals:

        for mode in ['percentiles', 'natural_breaks']:
            # ['percentiles','natural_breaks']
            for criterion in ['Vollzeitaequivalent']:
                # ['MedianIncome2017','CohabProxIndex','SENIOR_ANT',
                #  'Living_space_per_Person_2017','1PHouseholds',
                #  'Vollzeitaequivalent']
                for n in range(3, 6):
                    plot_area_timeseries(criterion, mode, n)
                    zone_gdf = assign_zones(criterion, mode, n)
                    tra_arr, zone_lst = load_model()
                    foot_df, bike_df, publ_df, moto_df \
                        = compute_mobility(tra_arr, zone_gdf, zone_lst, criterion, \
                                           mode, n)

    if bootstrap:
        criterion = 'random'

        for i in range(1, n_boots + 1):
            mode = 'tiles_' + str(i).zfill(3)

            plot_area_timeseries(criterion, mode, n)
            zone_gdf = assign_zones(criterion, mode, n)
            tra_arr, zone_lst = load_model()
            foot_df, bike_df, publ_df, moto_df \
                = compute_mobility(tra_arr, zone_gdf, zone_lst, criterion, \
                                   mode, n)

    return None


### FUNCTIONS #################################################################

def compile_total_ts(criterion, mode, n):
    name1 = 'bs_verkehrszaehldaten_velos_fussgaenger.geojson'
    name2 = 'bs_verkehrszaehldaten_motorisiert_individual.geojson'
    outname1 = name1[:-8] + '_' + criterion + '_' + str(n) + mode + '_assigned.geojson'
    outname2 = name2[:-8] + '_' + criterion + '_' + str(n) + mode + '_assigned.geojson'
    adm_gdf = load_areas(criterion, mode, n)

    if not os.path.isfile(tmsrs_dir + outname1):

        ts_gdf1 = load_timeseries(name1)

        # associate counts with areas
        dists_df1 = pd.DataFrame(columns=adm_gdf.percentile.unique(), \
                                 index=ts_gdf1.index)
        title = 'count-site association (foot/bike) for ' + str(n) + ' ' + \
                criterion + ' ' + ' '.join(mode.split('_'))
        for i in tqdm(dists_df1.index, title):
            for c in dists_df1.columns:
                AREA = adm_gdf.percentile == c
                point = ts_gdf1.loc[i, 'geometry']
                mpoly = adm_gdf.loc[AREA, 'geometry'].values[0]
                min_d = 10 ** 10
                for poly in list(mpoly):
                    if poly.contains(point):
                        min_d = 0.
                        break
                    else:
                        min_d = min(min_d, poly.exterior.distance(point))
                dists_df1.loc[i, c] = min_d

        assign_df1 = pd.DataFrame(dists_df1.astype(float).idxmin(axis=1))
        assign_df1.columns = ['percentile']

        ts_gdf1['percentile'] = assign_df1['percentile'].astype(int)
        ts_gdf1.to_file(tmsrs_dir + outname1, driver='GeoJSON')

    else:
        ts_gdf1 = gpd.read_file(tmsrs_dir + outname1)

    if not os.path.isfile(tmsrs_dir + outname2):

        ts_gdf2 = load_timeseries(name2)

        # associate counts with areas
        dists_df2 = pd.DataFrame(columns=adm_gdf.percentile.unique(), \
                                 index=ts_gdf2.index)
        title = 'count-site association (motorized) for ' + str(n) + ' ' + \
                criterion + ' ' + ' '.join(mode.split('_'))
        for i in tqdm(dists_df2.index, title):
            for c in dists_df2.columns:
                AREA = adm_gdf.percentile == c
                point = ts_gdf2.loc[i, 'geometry']
                mpoly = adm_gdf.loc[AREA, 'geometry'].values[0]
                min_d = 10 ** 10
                for poly in list(mpoly):
                    if poly.contains(point):
                        min_d = 0.
                        break
                    else:
                        min_d = min(min_d, poly.exterior.distance(point))
                dists_df2.loc[i, c] = min_d

        assign_df2 = pd.DataFrame(dists_df2.astype(float).idxmin(axis=1))
        assign_df2.columns = ['percentile']

        ts_gdf2['percentile'] = assign_df2['percentile'].astype(int)
        ts_gdf2.to_file(tmsrs_dir + outname2, driver='GeoJSON')

    else:
        ts_gdf2 = gpd.read_file(tmsrs_dir + outname2)

    ts_df1 = ts_gdf1[['date', 'traffictype', 'percentile', 'total']].groupby( \
        ['date', 'traffictype', 'percentile']).sum().reset_index()
    ts_df2 = ts_gdf2[['date', 'traffictype', 'percentile', 'total']].groupby( \
        ['date', 'traffictype', 'percentile']).sum().reset_index()
    ts_df = ts_df1.append(ts_df2).reset_index(drop=True)

    ts_df = ts_df[['date', 'total']].groupby('date').sum() \
        .reset_index().sort_values('date')

    for i in ts_df.index:
        ts_df.loc[i, 'date'] = datetime.strptime(ts_df.loc[i, 'date'], '%d.%m.%Y') \
            .date()

    A = ts_df.date >= datetime(2020, 2, 3).date()
    B = ts_df.date <= datetime(2020, 5, 24).date()
    ts_df = ts_df[A & B].sort_values('date').reset_index(drop=True)

    ts_df['week'] = ts_df.index // 7
    ts_df = ts_df.groupby('week').agg({'date': 'min', 'total': 'sum'})
    ts_df.date = ts_df.date + timedelta(days=3)

    # create figure
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111)
    plt.plot(ts_df.total / ts_df.total.max(), label='privat')

    # load BVB data
    os.chdir(sbbbvb_dir)
    df = pd.read_csv('BVB_Fahrgastzahlen.csv')
    df.columns = ['date', 'total', 'week']
    os.chdir(main_dir)

    df = df[['date', 'total']]
    for i in df.index:
        df.loc[i, 'date'] = datetime.strptime(df.loc[i, 'date'], '%Y-%m-%d') \
            .date()

    df.date = df.date + timedelta(days=3)
    df = df.sort_values('date').reset_index(drop=True)

    # add BVB and privat traffic timeseries by means of traffic-model weights
    tra_arr, zone_lst = load_model()
    ts_df.total = (ts_df.total / ts_df.total.max()) * tra_arr[[0, 1, 3], :, :].sum() + \
                  (df.total / df.total.max()) * tra_arr[2, :, :].sum()
    ts_df.total = ts_df.total / ts_df.total.max()
    print(ts_df)

    # add to figure
    ax1.set_xticks(ts_df.index)
    ax1.set_xticklabels(ts_df.date, rotation=45)
    plt.plot(df.total / df.total.max(), label='public')
    plt.plot(ts_df.total, label='total')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    os.chdir(figure_dir)
    fig.savefig('bs_full_traffic_model_timeseries.png', dpi=300)
    os.chdir(main_dir)

    # export
    os.chdir(output_dir)
    ts_df.to_csv('bs_full_traffic_model_timeseries.csv')
    os.chdir(main_dir)

    return ts_df


def load_model():
    '''creates an np.array of shape 4xnxn, where n is the number of
    zones of the traffic model of Basel Stadt. The first four dimensions
    refer to foot, bike, public and car transport, in that order.'''

    cols = ['a', 'b', 'v']

    os.chdir(trans_dir)
    foot_df = pd.read_csv('Fuss_DWV_aggr_gebiete_CbyC.CSV')
    bike_df = pd.read_csv('Velo_DWV_aggr_gebiete_CbyC.CSV')
    publ_df = pd.read_csv('OeV_DWV_aggr_gebiete_CbyC.CSV')
    moto_df = pd.read_csv('PWPersonen_DWV_aggr_gebiete_CbyC.CSV')
    os.chdir(main_dir)

    # number of zones is length of df minus 1, dut to omission of zone 0
    n_zones = len(foot_df[foot_df.columns[0]].unique()) - 1
    trans_arr = np.zeros(shape=(4, n_zones, n_zones))

    dfs = [foot_df, bike_df, publ_df, moto_df]
    for i, df in enumerate(dfs):
        df.columns = cols
        df = df[(df.a != 0) & (df.b != 0)]
        df = df.sort_values(['a', 'b'])
        trans_arr[i] = df.v.values.reshape(n_zones, n_zones)

    zone_lst = df.a.unique().tolist()

    # normalize such that mean of sum over all modes of transport is one
    trans_arr = trans_arr / trans_arr.sum(axis=0).mean()

    return trans_arr, zone_lst


def load_areas(criterion, mode, n):
    os.chdir('/'.join(shape_dir.split('/')[:-3]) + '/derived/')
    title = 'bs_' + criterion + '_' + str(n) + mode + '.shp'
    gdf = gpd.read_file(title)
    gdf = gdf.to_crs(CRS('epsg:2056'))
    os.chdir(main_dir)

    return gdf


def assign_zones(criterion, mode, n):
    adm_gdf = load_areas(criterion, mode, n)
    tra_arr, zone_lst = load_model()

    folder = zone_dir + '01_Zonen/'
    name = 'Zonen_Centroid'

    os.chdir(folder)
    zone_gdf = gpd.read_file(name + '.shp')
    zone_gdf = zone_gdf.to_crs(CRS('epsg:2056'))
    os.chdir(main_dir)

    # assign stations to closest socio-economic area
    dists_df = pd.DataFrame(columns=adm_gdf.percentile.unique(), \
                            index=zone_lst)
    title = 'zone-area association for ' + str(n) + ' ' + criterion + ' ' + \
            ' '.join(mode.split('_'))
    for i in tqdm(dists_df.index, title):
        for c in dists_df.columns:
            AREA = adm_gdf.percentile == c
            point = zone_gdf.loc[zone_gdf.ID == i, 'geometry'].values[0]
            mpoly = adm_gdf.loc[AREA, 'geometry'].values[0]
            min_d = 10 ** 10
            for poly in list(mpoly):
                if poly.contains(point):
                    min_d = 0.
                    break
                else:
                    min_d = min(min_d, poly.exterior.distance(point))
            dists_df.loc[i, c] = min_d

    assign_df = pd.DataFrame(dists_df.astype(float).idxmin(axis=1))
    assign_df.columns = ['percentile']

    for i in zone_gdf.index:
        z = zone_gdf.loc[i, 'ID']
        if z in assign_df.index:
            zone_gdf.loc[i, 'percentile'] = assign_df.loc[z, 'percentile']
    zone_gdf = zone_gdf[zone_gdf.percentile.notnull()]
    zone_gdf.percentile = zone_gdf.percentile.astype(int)

    # plot zone assignment
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('traffic-zone assignment : ' + str(n) + ' ' + ' '.join(mode.split('_')))
    plt.tight_layout()

    zone_gdf[zone_gdf.percentile.notnull()].to_crs(CRS('epsg:2056')) \
        .plot(ax=ax1, zorder=10, markersize=10, column='percentile', legend=True, \
              categorical=True, cmap='Spectral')

    os.chdir(figure_dir)
    fig.savefig('bs_' + criterion + '_' + str(n) + mode + '_traffic_zones.png', dpi=300)
    os.chdir(main_dir)

    return zone_gdf


def compute_mobility(tra_arr, zone_gdf, zone_lst, criterion, mode, n):
    adm_gdf = load_areas(criterion, mode, n)

    foot_df = pd.DataFrame(tra_arr[0], index=zone_lst, columns=zone_lst)
    bike_df = pd.DataFrame(tra_arr[1], index=zone_lst, columns=zone_lst)
    publ_df = pd.DataFrame(tra_arr[2], index=zone_lst, columns=zone_lst)
    moto_df = pd.DataFrame(tra_arr[3], index=zone_lst, columns=zone_lst)

    percentiles = zone_gdf.percentile.unique().tolist()
    percentiles.sort()
    percentiles = np.array(percentiles)

    foot_rdf = pd.DataFrame(index=percentiles, columns=percentiles)
    bike_rdf = pd.DataFrame(index=percentiles, columns=percentiles)
    publ_rdf = pd.DataFrame(index=percentiles, columns=percentiles)
    moto_rdf = pd.DataFrame(index=percentiles, columns=percentiles)

    for a in percentiles:
        A = zone_gdf.percentile == a
        a_ids = zone_gdf[A].ID.tolist()
        if criterion == 'random':
            a_area = adm_gdf.loc[a - 1, 'geometry'].area
        else:
            a_area = adm_gdf.loc[a, 'geometry'].area
        for b in percentiles[percentiles >= a]:
            B = zone_gdf.percentile == b
            b_ids = zone_gdf[B].ID.tolist()
            if criterion == 'random':
                b_area = adm_gdf.loc[b - 1, 'geometry'].area
            else:
                b_area = adm_gdf.loc[b, 'geometry'].area
            area = a_area + b_area
            foot_rdf.loc[a, b] = foot_df.loc[a_ids, b_ids].sum().sum()  # /area
            bike_rdf.loc[a, b] = bike_df.loc[a_ids, b_ids].sum().sum()  # /area
            publ_rdf.loc[a, b] = publ_df.loc[a_ids, b_ids].sum().sum()  # /area
            moto_rdf.loc[a, b] = moto_df.loc[a_ids, b_ids].sum().sum()  # /area
            foot_rdf.loc[b, a] = foot_df.loc[a_ids, b_ids].sum().sum()  # /area
            bike_rdf.loc[b, a] = bike_df.loc[a_ids, b_ids].sum().sum()  # /area
            publ_rdf.loc[b, a] = publ_df.loc[a_ids, b_ids].sum().sum()  # /area
            moto_rdf.loc[b, a] = moto_df.loc[a_ids, b_ids].sum().sum()  # /area

    # export mobility matrices
    os.chdir(graph_dir)
    foot_rdf.to_csv('bs_' + criterion + '_' + str(n) + mode + '_foot_mobility.csv')
    bike_rdf.to_csv('bs_' + criterion + '_' + str(n) + mode + '_bike_mobility.csv')
    publ_rdf.to_csv('bs_' + criterion + '_' + str(n) + mode + '_publ_mobility.csv')
    moto_rdf.to_csv('bs_' + criterion + '_' + str(n) + mode + '_moto_mobility.csv')
    os.chdir(main_dir)

    # create graphs
    grp = ig.Graph()
    if criterion == 'random':
        grp.add_vertices(foot_rdf.shape[0])
        grp.vs['name'] = foot_rdf.index
    else:
        grp.add_vertices(foot_rdf.shape[0] - 1)
        grp.vs['name'] = foot_rdf.index[1:].tolist()

    m = (foot_rdf + bike_rdf + publ_rdf + moto_rdf).mean().mean() / 40.
    w = []
    if criterion == 'random':
        for i in foot_rdf.index:
            for j in foot_rdf.index:
                grp.add_edges([(i - 1, j - 1), (i - 1, j - 1), (i - 1, j - 1), (i - 1, j - 1)])
                w += [.5 * (foot_rdf.loc[i, j] + foot_rdf.loc[j, i]) / m, \
                      .5 * (bike_rdf.loc[i, j] + bike_rdf.loc[j, i]) / m, \
                      .5 * (publ_rdf.loc[i, j] + publ_rdf.loc[j, i]) / m, \
                      .5 * (moto_rdf.loc[i, j] + moto_rdf.loc[j, i]) / m]
    else:
        for i in foot_rdf.index[1:]:
            for j in foot_rdf.index[i:]:
                grp.add_edges([(i - 1, j - 1), (i - 1, j - 1), (i - 1, j - 1), (i - 1, j - 1)])
                w += [.5 * (foot_rdf.loc[i, j] + foot_rdf.loc[j, i]) / m, \
                      .5 * (bike_rdf.loc[i, j] + bike_rdf.loc[j, i]) / m, \
                      .5 * (publ_rdf.loc[i, j] + publ_rdf.loc[j, i]) / m, \
                      .5 * (moto_rdf.loc[i, j] + moto_rdf.loc[j, i]) / m]

    grp.es['color'] = foot_rdf.shape[0] * ['seagreen', 'steelblue', 'coral', 'crimson']
    grp.es['width'] = w

    # plot graphs
    os.chdir(figure_dir)

    visual_style = {}
    visual_style['vertex_size'] = 40
    visual_style['vertex_color'] = 'gray'
    visual_style['vertex_label'] = grp.vs['name']
    visual_style['vertex_label_size'] = 20
    visual_style['layout'] = grp.layout('kk')
    visual_style['bbox'] = (600, 600)
    visual_style['margin'] = 100

    grp_fig = ig.Plot('bs_' + criterion + '_' + str(n) + mode + '_traffic_graphs.png', \
                      background='white')

    grp_fig.add(grp, **visual_style)
    grp_fig.redraw()
    grp_fig.save()

    os.chdir(main_dir)

    return foot_rdf, bike_rdf, publ_rdf, moto_rdf


def load_timeseries(name):
    os.chdir(tmsrs_dir)
    gdf = gpd.read_file(name)
    os.chdir(main_dir)

    gdf = gdf[['date', 'sitecode', 'traffictype', 'total', 'geometry']].dissolve( \
        by=['date', 'sitecode', 'traffictype'], aggfunc='sum').reset_index()

    gdf = gdf.to_crs(CRS('epsg:2056'))

    return gdf


def plot_area_timeseries(criterion, mode, n):
    name1 = 'bs_verkehrszaehldaten_velos_fussgaenger.geojson'
    name2 = 'bs_verkehrszaehldaten_motorisiert_individual.geojson'
    outname1 = name1[:-8] + '_' + criterion + '_' + str(n) + mode + '_assigned.geojson'
    outname2 = name2[:-8] + '_' + criterion + '_' + str(n) + mode + '_assigned.geojson'
    adm_gdf = load_areas(criterion, mode, n)

    ts_gdf1 = load_timeseries(name1)

    # associate counts with areas
    dists_df1 = pd.DataFrame(columns=adm_gdf.percentile.unique(), \
                             index=ts_gdf1.index)
    title = 'count-site association (foot/bike) for ' + str(n) + ' ' + \
            criterion + ' ' + ' '.join(mode.split('_'))
    for i in tqdm(dists_df1.index, title):
        for c in dists_df1.columns:
            AREA = adm_gdf.percentile == c
            point = ts_gdf1.loc[i, 'geometry']
            mpoly = adm_gdf.loc[AREA, 'geometry'].values[0]
            min_d = 10 ** 10
            for poly in list(mpoly):
                if poly.contains(point):
                    min_d = 0.
                    break
                else:
                    min_d = min(min_d, poly.exterior.distance(point))
            dists_df1.loc[i, c] = min_d

    assign_df1 = pd.DataFrame(dists_df1.astype(float).idxmin(axis=1))
    assign_df1.columns = ['percentile']

    ts_gdf1['percentile'] = assign_df1['percentile'].astype(int)
    ts_gdf1.to_file(tmsrs_dir + outname1, driver='GeoJSON')

    ts_gdf2 = load_timeseries(name2)

    # associate counts with areas
    dists_df2 = pd.DataFrame(columns=adm_gdf.percentile.unique(), \
                             index=ts_gdf2.index)
    title = 'count-site association (motorized) for ' + str(n) + ' ' + \
            criterion + ' ' + ' '.join(mode.split('_'))
    for i in tqdm(dists_df2.index, title):
        for c in dists_df2.columns:
            AREA = adm_gdf.percentile == c
            point = ts_gdf2.loc[i, 'geometry']
            mpoly = adm_gdf.loc[AREA, 'geometry'].values[0]
            min_d = 10 ** 10
            for poly in list(mpoly):
                if poly.contains(point):
                    min_d = 0.
                    break
                else:
                    min_d = min(min_d, poly.exterior.distance(point))
            dists_df2.loc[i, c] = min_d

    assign_df2 = pd.DataFrame(dists_df2.astype(float).idxmin(axis=1))
    assign_df2.columns = ['percentile']

    ts_gdf2['percentile'] = assign_df2['percentile'].astype(int)
    ts_gdf2.to_file(tmsrs_dir + outname2, driver='GeoJSON')

    ts_gdf1 = ts_gdf1[['date', 'traffictype', 'percentile', 'total']].groupby( \
        ['date', 'traffictype', 'percentile']).sum().reset_index()
    ts_gdf2 = ts_gdf2[['date', 'traffictype', 'percentile', 'total']].groupby( \
        ['date', 'traffictype', 'percentile']).sum().reset_index()
    ts_gdf = ts_gdf1.append(ts_gdf2).reset_index(drop=True)

    for i in ts_gdf.index:
        ts_gdf.loc[i, 'date'] = datetime.strptime(ts_gdf.loc[i, 'date'], '%d.%m.%Y') \
            .date()

    # create figure
    fig = plt.figure(figsize=(20, 15))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    cls = mpl.cm.viridis(np.linspace(0, .9, n))
    for p in range(1, n + 1):
        P = ts_gdf.percentile == p
        T1 = ts_gdf.traffictype == 'Fussgänger'
        T2 = ts_gdf.traffictype == 'Velo'
        T3 = ts_gdf.traffictype == 'MIV'
        plot_df1 = ts_gdf[P & T1].sort_values('date').reset_index()
        plot_df2 = ts_gdf[P & T2].sort_values('date').reset_index()
        plot_df3 = ts_gdf[P & T3].sort_values('date').reset_index()
        plot_df4 = ts_gdf.loc[P, ['date', 'total']].groupby('date').sum()
        plot_df4 = plot_df4.sort_values('date').reset_index()
        ax1.plot(plot_df1.total, c=cls[p - 1], label=str(p))
        ax2.plot(plot_df2.total, c=cls[p - 1], label=str(p))
        ax3.plot(plot_df3.total, c=cls[p - 1], label=str(p))
        ax4.plot(plot_df4.total, c=cls[p - 1], label=str(p))
    ax1.set_title('Pass-by Foot Mobility by Area')
    ax2.set_title('Pass-by Bike Mobility by Area')
    ax3.set_title('Pass-by Car Mobility by Area')
    ax4.set_title('Pass-by Total Mobility by Area')
    xs = [0, 15, 29, 44, 60, 75, 89]
    ls = ts_gdf.date.unique()[xs]  # plot_df4.loc[plot_df4.index[xs],'date']
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks(xs)
        ax.set_xticklabels(ls, rotation=45, horizontalalignment='right', fontsize=8)
        ax.legend()
    plt.tight_layout()

    os.chdir(figure_dir)
    figname = name1[:-26] + '_' + criterion + '_' + str(n) + mode + '_timeseries.png'
    fig.savefig(figname, dpi=200)
    os.chdir(main_dir)
    plt.close()

    return ts_gdf


###############################################################################

if __name__ == '__main__':
    main()

###############################################################################
