#!user/bin/env python3
#
# Collection of routines to partition Basel Stadt into socio-economic areas
# which subsequently may be employed to be equiped with a transport graph
# structure etc.
#
### PACKAGES ##################################################################

import cairo                as cr
import descartes
import dictances            as dist
import geopandas            as gpd
import igraph               as ig
import math
import matplotlib.pyplot    as plt
import numpy                as np
import os
import pandas               as pd
import wquantiles           as wq
import shapely.geometry     as geometry
from jenkspy import jenks_breaks
from pyproj import CRS
from shapely.geometry import Point
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
from igraph.drawing.text import TextDrawer
# from   longsgis             import voronoiDiagram4plg as polyvor
from tqdm import tqdm

### GLOBAL VARIABLES ##########################################################

main_dir = os.getcwd()
data_dir = 'geodata/'
shape_dir = '../../../data/shapefiles/GeodatenBS_20200605/'
graph_dir = 'graphs/'
figure_dir = '../../../figures/'
out_dir = 'output/'


### MAIN ######################################################################

def main():
    partition = True
    bootstrap = False
    pca = False

    if partition:
        for mode in ['percentiles', 'natural_breaks']:
            for criterion in ['Vollzeitaequivalent']:
                # ['MedianIncome2017','CohabProxIndex','SENIOR_ANT',\
                #  'Living space per Person 2017','1PHouseholds',\
                #  'Vollzeitaequivalent']:
                title = 'partitioning wrt ' + criterion + ' ' + ' '.join(mode.split('_'))
                for n in tqdm(range(3, 10), title):
                    socio_df = build_areas(n=n, criterion=criterion, mode=mode)

    if bootstrap:
        n_tiles = 3
        n_partitions = 33
        bootstrap_partitions(n_tiles, n_partitions)

    if pca:
        indicator_lst = ['SENIOR_ANT', '1PHouseholds', \
                         'Living space per Person 2017']
        trans_df = indicator_pca(indicator_lst)

    return None


### FUNCTIONS #################################################################

def indicator_pca(indicator_lst):
    # load and merge
    ind_df = pd.DataFrame(columns=['BlockID'])
    for indicator in indicator_lst:
        df = load_data(indicator)
        df.rename(columns={'BLO_ID': 'BlockID'}, inplace=True)
        ind_df = pd.merge(ind_df, df, on='BlockID', how='outer')

    # set negative values to nans and remove nan rows
    ind_df[ind_df == -1] = np.nan
    ind_df.dropna(inplace=True)

    # normalize
    ind_df.set_index('BlockID', inplace=True, drop=True)
    ind_df = ind_df / ind_df.max()

    # perform pca
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(ind_df.values)
    pcs_df = pd.DataFrame(data=pcs, columns=['PC1', 'PC2', 'PC3'])
    trans_df = pd.DataFrame(pca.components_, \
                            index=pcs_df.columns, columns=ind_df.columns)
    trans_df['explained_variance'] = pca.explained_variance_ratio_

    return trans_df


def bootstrap_partitions(n_tiles, n_partitions):
    for i in range(n_partitions):
        # load n-percentile partition
        block_gdf = load_shp('Block')
        block_gdf.BLO_ID = block_gdf.BLO_ID.astype(int)
        criterion = 'MedianIncome2017'  # could be any criterion
        socio_df = load_data(criterion)
        socio_gdf = pd.merge(block_gdf, socio_df, \
                             left_on='BLO_ID', \
                             right_on='BlockID', \
                             how='outer')
        cols = ['BLO_ID', 'WOV_ID', criterion, 'geometry']
        socio_gdf = socio_gdf[cols]

        # randomly assign percentiles
        socio_gdf['percentile'] = np.random.choice(range(1, n_tiles + 1), \
                                                   size=socio_gdf.shape[0])
        socio_gdf.percentile = socio_gdf.percentile.astype(int)
        socio_df = pd.DataFrame(socio_gdf.drop('geometry', axis=1), copy=True)
        print('\n partition ' + str(i + 1) + ':')
        print(socio_df.percentile.describe())

        # join blocks of same percentile to one multipolygon
        socio_gdf.loc[:, 'geometry'] = socio_gdf.loc[:, 'geometry'].buffer(0)
        socio_gdf = socio_gdf[['percentile', 'geometry']] \
            .dissolve(by='percentile')
        socio_gdf.reset_index(inplace=True)
        socio_gdf.rename(columns={'index': 'percentile'}, inplace=True)

        # visualize
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        socio_gdf.plot(column='percentile', categorical=True, cmap='Spectral', \
                       ax=ax1, legend=True)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title('random ' + str(n_tiles) + '-area partition')
        plt.tight_layout()
        plt.close()

        # export
        title = 'bs_random_' + str(n_tiles) + 'tiles_' + str(i + 1).zfill(3)
        os.chdir(figure_dir)
        fig.savefig(title + '.png', dpi=300)
        os.chdir(main_dir)
        os.chdir('/'.join(shape_dir.split('/')[:-2]) + '/derived/')
        socio_gdf.to_file(title + '.shp')
        os.chdir(main_dir)
        os.chdir(graph_dir)
        socio_df.to_csv(title + '.csv')
        os.chdir(main_dir)

    return None


def build_areas(n, criterion, mode):
    # load socio-economic data and block shapes
    block_gdf = load_shp('Block')
    block_gdf.BLO_ID = block_gdf.BLO_ID.astype(int)
    socio_df = load_data(criterion)

    if criterion in ['Living space per Person 2017', 'MedianIncome2017']:
        socio_gdf = pd.merge(block_gdf, socio_df, \
                             left_on='BLO_ID', \
                             right_on='BlockID', \
                             how='outer')
    elif criterion in ['SENIOR_ANT', 'CohabProxIndex', '1PHouseholds', \
                       'Vollzeitaequivalent']:
        socio_gdf = pd.merge(block_gdf, socio_df, \
                             on='BLO_ID', \
                             how='outer')

    cols = ['BLO_ID', 'WOV_ID', criterion, 'geometry']
    socio_gdf = socio_gdf[cols]

    # calculate and assign percentiles / jenks natural breaks
    socio_gdf.loc[socio_gdf[criterion] < 0, criterion] = np.nan
    if mode == 'percentiles':
        percs = np.percentile(socio_gdf[criterion].dropna(), \
                              np.linspace(0, 100, n + 1))
    elif mode == 'natural_breaks':
        percs = jenks_breaks(socio_gdf[criterion].dropna(), nb_class=n)

    for i, p in enumerate(percs[:-1]):
        COND = socio_gdf[criterion] >= p
        socio_gdf.loc[COND, 'percentile'] = i + 1

    COND = np.isnan(socio_gdf[criterion])
    socio_gdf.loc[COND, 'percentile'] = 0

    socio_gdf.percentile = socio_gdf.percentile.astype(int)
    socio_df = pd.DataFrame(socio_gdf.drop('geometry', axis=1), copy=True)

    # join blocks of same percentile to one multipolygon
    socio_gdf.loc[:, 'geometry'] = socio_gdf.loc[:, 'geometry'].buffer(0)
    socio_gdf = socio_gdf[['percentile', 'geometry']].dissolve(by='percentile')
    socio_gdf.reset_index(inplace=True)
    socio_gdf.rename(columns={'index': 'percentile'}, inplace=True)

    # visualize
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    socio_gdf.plot(column='percentile', categorical=True, cmap='Spectral', ax=ax1, \
                   legend=True)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(criterion + ' : ' + str(n) + ' ' + ' '.join(mode.split('_')))
    plt.tight_layout()
    plt.close()

    os.chdir(figure_dir)
    fig.savefig('bs_' + '_'.join(criterion.split(' ')) + '_' + str(n) + mode + '.png', \
                dpi=300)
    os.chdir(main_dir)

    # export
    os.chdir('/'.join(shape_dir.split('/')[:-2]) + '/derived/')
    socio_gdf.to_file('bs_' + '_'.join(criterion.split(' ')) + '_' + str(n) + mode + '.shp')
    os.chdir(main_dir)
    os.chdir(graph_dir)
    socio_df.to_csv('bs_' + '_'.join(criterion.split(' ')) + '_' + str(n) + mode + '.csv')
    os.chdir(main_dir)

    return socio_df


def load_data(criterion):
    os.chdir(data_dir)
    if criterion == 'Living space per Person 2017':
        df = pd.read_csv('SocioeconomicScore_data.csv')
        df = df[['BlockID', criterion]]
    elif criterion == 'SENIOR_ANT':
        df = pd.read_csv('Bevoelkerung.csv')
        df = df[['BLO_ID', criterion]]
    elif criterion == 'CohabProxIndex':
        df = pd.read_csv('CohabProxIndex.csv')
        df = df[['BLO_ID', criterion]]
    elif criterion == 'MedianIncome2017':
        df = pd.read_csv('MedianIncome2017.csv')
        df = df[['BlockID', criterion]]
    elif criterion == '1PHouseholds':
        df = pd.read_csv('1PHouseholds.csv')
        df = df[['BLO_ID', criterion]]
    elif criterion == 'Vollzeitaequivalent':
        df = pd.read_csv('Vollzeitaequivalent.csv')
        df = df[['BLO_ID', criterion]]

    os.chdir(main_dir)

    return df


def load_shp(data_type):
    os.chdir(shape_dir + data_type)
    gdf = gpd.read_file(data_type + '.shp')
    os.chdir(main_dir)

    return gdf


###############################################################################

if __name__ == "__main__":
    df = main()
