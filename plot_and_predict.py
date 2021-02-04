#!/usr/bin/env python3
#
# This is script contains the evalaution and prediction functions

### PACKAGES ###################################################################

import datetime          as dt
# import igraph            as ig
import matplotlib.pyplot as plt
import numpy             as np
import os
import pandas            as pd
import pickle
import random
import shelve
import sklearn
from joblib import Parallel, delayed
# import jenkspy

# from IPython import embed
from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

### GLOBAL VARIABLE ############################################################
global constantMobility
constantMobility = False

global zeroMobility
zeroMobility = False

global constantR
constantR = False

global fixedSocial
fixedSocial = 1

global time_dep_soc
global randomIndex
randomIndex = '001'


# Paths and filenames to be included by user
global filenameCaseData
filenameCaseData = 'test.csv'

global filenameKalmanData
filenameKalmanData = 'test.csv'

global filenameSocioeconomicData
filenameSocioeconomicData = 'test.csv'

global filenameMobilityData
filenameMobilityData = 'test.csv'

# Mobility graphs for each mode of transport of 'publ', 'bike', 'moto', 'foot' - one csv file each
global filenameSocioeconomicGraphData
filenameSocioeconomicGraphData = 'test'

# Add all result files
global result_file
result_file   = 'results_test.pkl'

global fixedPar_file
fixedPar_file = 'fixedPars_test.pkl'




def main():
    # Get social time series
    df_timedep = pd.read_csv(filenameMobilityData)

    # Get social time series
    df_Kalman   = pd.read_csv(filenameKalmanData)
    time_Kalman = df_Kalman['timestamp'].values
    R_estimate  = df_Kalman['R_estimate'].values


    global alpha_mob
    alpha_mob = time_dep(time_Kalman)
    y_Kalman = R_estimate / alpha_mob
    y_soc = y_Kalman / np.max(y_Kalman)

    global time_dep_soc
    time_dep_soc = UnivariateSpline(time_Kalman, y_soc, s=0.05)


    # Loop over all runIDs
    runIDs = ['yourRUNID_1', 'yourRUNID2']
    names = ['NameRUNID_1','NameRUNID_2']

    for i, id in enumerate(runIDs):
        evalVaccination([id], [names[i]])
        evalVaccination_ICUoccupation([id], [names[i]])
        plot_results(id)

    return 0


def evalVaccination_ICUoccupation(runIDs, names):

    # Parameters to set
    vaccineGroup = 3                    # select Tertile
    fractions_vaccGroup = [0.3333333]   # Fraction of population vaccinated
    effVacsGroup        = [0.9]         # Vaccine efficacy
    icu_stay     = 5.9                  # Icu stay duration
    icu_capa           = 44             # ICU capacity
    icu_perc = 0.01                     # Propability for all infected cases to be on ICU
    all_quat = [1, 2, 3]                # the summarized quaters of Basel to be analysed: choose from 1-9,'all'
    vaccinateALLsenior = True           # If all senior citizens were vaccinated reduce ICU propability
    if vaccinateALLsenior:
        icu_perc2 = 0.005
    else:
        icu_perc2 = 0.01
    t = list(np.arange(0, 150))         # Simualte for 150 days


    # Plot colors
    colors = ['black', 'darkblue', 'gold', 'royalblue', 'orange', 'yellow']

    # Global parameters
    global constantR
    global constantMobility
    global zeroMobility

    fig1 = plt.figure(figsize=(4, 4))
    ax = fig1.add_subplot(1, 1, 1)
    counter = 0
    for i_q, run_ID in enumerate(runIDs):

        # load data and fit results for this run
        infile = open(result_file, 'rb')
        result = pickle.load(infile)
        infile.close()

        # load fixed parameter data
        infile = open(fixedPar_file, 'rb')
        fixed_pars = pickle.load(infile)
        infile.close()

        # Prepare parameters
        n_tot        = np.sum(fixed_pars[6])
        n_vaccinated = fixed_pars[6][vaccineGroup[i_q] - 1]
        f25          = fractions_vaccGroup[0] * n_tot / n_vaccinated
        fractions_vacc_group = [f25]
        fractions_vacc_all   = [0, fractions_vaccGroup[0]]

        for ev in range(0, len(effVacsGroup)):
            effVac = effVacsGroup[ev]

            if counter == 0:

                # random distribution of vaccinated subjects
                for i_fv, frac_vac in enumerate(fractions_vacc_all):

                    if frac_vac == 0 and ev > 0:
                        continue

                    n_inf = predict_Vaccine(result[0].x, fixed_pars, 0, t, 1, effVac)

                    mean_ninf = np.zeros((n_inf.shape[0],))
                    min_ninf = np.zeros((n_inf.shape[0],))
                    max_ninf = np.zeros((n_inf.shape[0],))
                    for j in range(0, len(result)):
                        if np.isin(j, inds_skip):
                            continue

                        for q in all_quat:
                            if q == all_quat[0]:
                                mdl_cuminf_uncer = predict_Vaccine(result[j].x, fixed_pars, frac_vac, t, q, effVac)
                            else:
                                mdl_cuminf_uncer = mdl_cuminf_uncer + predict_Vaccine(result[j].x, fixed_pars, frac_vac,
                                                                                      t, q, effVac)

                        mean_ninf = mean_ninf + mdl_cuminf_uncer

                        if j > 0:
                            min_ninf = np.min(np.vstack((min_ninf, mdl_cuminf_uncer)), axis=0)
                            max_ninf = np.max(np.vstack((max_ninf, mdl_cuminf_uncer)), axis=0)
                        else:
                            min_ninf = mdl_cuminf_uncer
                            max_ninf = mdl_cuminf_uncer

                    # Average over boot straps
                    mean_ninf = mean_ninf / (len(result) - len(inds_skip))
                    if frac_vac == 0:

                        yV0_int = interp1d(t, mean_ninf)
                        cases = []
                        for it in range(0, len(t) - int(round(icu_stay))):
                            cases.append((yV0_int(it + icu_stay) - yV0_int(it)) * icu_perc)

                        yV0_int_cases = interp1d(100 * np.array(cases)[:cases.index(max(cases))] / icu_capa,
                                                 np.arange(cases.index(max(cases))))
                        yV0 = 50  #
                        xV0 = yV0_int_cases(50)

                        ax.plot(np.arange(len(cases)), 100 * np.array(cases) / icu_capa, '-', linewidth=2,
                                color=colors[counter], zorder=12, label='V0 - no vaccine')

                    else:

                        yV1_int = interp1d(t, mean_ninf)
                        cases = []
                        for it in range(0, len(t) - int(round(icu_stay))):
                            cases.append((yV1_int(it + icu_stay) - yV1_int(it)) * icu_perc)

                        yV1_int_cases = interp1d(100 * np.array(cases)[:cases.index(max(cases))] / icu_capa,
                                                 np.arange(cases.index(max(cases))))
                        yV1 = 50  #
                        xV1 = yV1_int_cases(50)
                        ax.plot(np.arange(len(cases)), 100 * np.array(cases) / icu_capa, '-', linewidth=2,
                                color=colors[counter], zorder=12,
                                label='V1 - ' + f"{frac_vac * 100:.0f}" + '% randomly')  # with '+ f"{effVac*100:.0f}"+'% efficacy')

                    cases_min = []
                    cases_max = []
                    yV0_int_min = interp1d(t, min_ninf)
                    yV0_int_max = interp1d(t, max_ninf)
                    for it in range(0, len(t) - int(round(icu_stay))):
                        cases_min.append((yV0_int_min(it + icu_stay) - yV0_int_min(it)) * icu_perc)
                        cases_max.append((yV0_int_max(it + icu_stay) - yV0_int_max(it)) * icu_perc)

                    ax.plot(np.arange(len(cases_min)), 100 * np.array(cases_min) / icu_capa, '--',
                            color=colors[counter], zorder=1, alpha=0.5, linewidth=1)
                    ax.plot(np.arange(len(cases_min)), 100 * np.array(cases_max) / icu_capa, '--',
                            color=colors[counter], zorder=1, alpha=0.5, linewidth=1)
                    ax.fill_between(np.arange(len(cases_min)), 100 * np.array(cases_min) / icu_capa,
                                    100 * np.array(cases_max) / icu_capa, color=colors[counter], zorder=3, alpha=0.25,
                                    linewidth=0)

                    counter += 1

            # Vaccine per group
            for i_fv, frac_vac in enumerate(fractions_vacc_group):

                if frac_vac == 0 and ev > 0:
                    continue

                mean_ninf = np.zeros((len(t),))
                min_ninf = np.zeros((len(t),))
                max_ninf = np.zeros((len(t),))
                for j in range(0, len(result)):
                    if np.isin(j, inds_skip):
                        continue

                    for q in all_quat:
                        if q == all_quat[0]:
                            mdl_cuminf_uncer = predict_VaccineOneGroup(result[j].x, fixed_pars, frac_vac, t, q,
                                                                       vaccineGroup[i_q], effVac)
                        else:
                            mdl_cuminf_uncer = mdl_cuminf_uncer + predict_VaccineOneGroup(result[j].x, fixed_pars,
                                                                                          frac_vac, t, q,
                                                                                          vaccineGroup[i_q], effVac)

                    mean_ninf = mean_ninf + mdl_cuminf_uncer

                    if j > 0:
                        min_ninf = np.min(np.vstack((min_ninf, mdl_cuminf_uncer)), axis=0)
                        max_ninf = np.max(np.vstack((max_ninf, mdl_cuminf_uncer)), axis=0)
                    else:
                        min_ninf = mdl_cuminf_uncer
                        max_ninf = mdl_cuminf_uncer

                # Average over boot straps
                mean_ninf = mean_ninf / (len(result) - len(inds_skip))

                # Get 50% ICU occupancy mark
                yV2_int = interp1d(t, mean_ninf)
                cases = []
                for it in range(0, len(t) - int(round(icu_stay))):
                    cases.append((yV2_int(it + icu_stay) - yV2_int(it)) * icu_perc2)

                yV2_int_cases = interp1d(100 * np.array(cases)[:cases.index(max(cases))] / icu_capa,
                                         np.arange(cases.index(max(cases))))
                yV2 = 50  #
                xV2 = yV2_int_cases(50)

                # Plot
                if frac_vac == 0:
                    ax.plot(np.arange(len(cases)), 100 * np.array(cases) / icu_capa, '-', linewidth=2,
                            color=colors[counter], zorder=12, label='V0 - no vaccine')
                else:
                    if i_fv == 1:
                        if vaccinateALLsenior:
                            ax.plot(np.arange(len(cases)), 100 * np.array(cases) / icu_capa, '-', linewidth=2,
                                    color=colors[counter], zorder=12,
                                    label='V3 - ' + f"{frac_vac * 100:.0f}" + '% T' + str(vaccineGroup[i_q]) + ' ' +
                                          names[i_q] + '=' + f"{fractions_vaccGroup[0] * 100:.0f}" + '% total')

                        else:
                            ax.plot(np.arange(len(cases)), 100 * np.array(cases) / icu_capa, '-', linewidth=2,
                                    color=colors[counter], zorder=12,
                                    label='V2 - ' + f"{frac_vac * 100:.0f}" + '% T' + str(vaccineGroup[i_q]) + ' ' +
                                          names[i_q] + '=' + f"{fractions_vaccGroup[0] * 100:.0f}" + '% total')

                    else:
                        ax.plot(np.arange(len(cases)), 100 * np.array(cases) / icu_capa, '-', linewidth=2, color='red',
                                zorder=12,
                                label='V3 - ' + f"{fractions_vaccGroup[0] * 100:.0f}" + '% from T' + str(
                                    vaccineGroup[i_q]) + "\n" + names[i_q])

                # Bootstrap bands
                cases_min = []
                cases_max = []
                yV0_int_min = interp1d(t, min_ninf)
                yV0_int_max = interp1d(t, max_ninf)
                for it in range(0, len(t) - int(round(icu_stay))):
                    cases_min.append((yV0_int_min(it + icu_stay) - yV0_int_min(it)) * icu_perc2)
                    cases_max.append((yV0_int_max(it + icu_stay) - yV0_int_max(it)) * icu_perc2)
                ax.fill_between(np.arange(len(cases_min)), 100 * np.array(cases_min) / icu_capa,
                                100 * np.array(cases_max) / icu_capa, color='red', zorder=3, alpha=0.25, linewidth=0)
                ax.plot(np.arange(len(cases_min)), 100 * np.array(cases_min) / icu_capa, '--',
                        color='red', zorder=1, alpha=0.5, linewidth=1)
                ax.plot(np.arange(len(cases_min)), 100 * np.array(cases_max) / icu_capa, '--',
                        color='red', zorder=1, alpha=0.5, linewidth=1)

                counter += 1

    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.xlabel('Time [days]', fontsize=14)
    plt.legend(prop={'size': 10}, frameon=True, facecolor='lightgrey', loc='lower right')
    plt.ylim([0.01, 1000])
    plt.xlim([0, len(cases_min)])
    plt.yscale('log')
    plt.ylabel('ICU Occupancy [%]', fontsize=14)
    ax.patch.set_facecolor('lightgrey')
    plt.grid(color='white')
    plt.scatter(xV0, yV0, color='black', marker='o', s=50, zorder=20)
    plt.scatter(xV1, yV1, color='darkblue', marker='o', s=50, zorder=20)
    plt.scatter(xV2, yV2, color='red', marker='o', s=50, zorder=20)
    plt.tight_layout()

    return 0

def evalVaccination(runIDs, names):


    # Parameters to set
    vaccineGroup = 3                    # select Tertile
    fractions_vaccGroup = [0.3333333]   # Fraction of population vaccinated
    effVacsGroup = [0.9]                # Vaccine efficacy
    icu_stay = 5.9                      # Icu stay duration
    icu_capa = 44                       # ICU capacity
    icu_perc = 0.01                     # Propability for all infected cases to be on ICU
    all_quat = [1, 2, 3]                # the summarized quaters of Basel to be analysed: choose from 1-9,'all'
    vaccinateALLsenior = True           # If all senior citizens were vaccinated reduce ICU propability
    if vaccinateALLsenior:
        icu_perc2 = 0.005
    else:
        icu_perc2 = 0.01
    t = list(np.arange(0, 150))         # Simualte for 150 days


    colors = ['black', 'darkblue', 'gold', 'royalblue', 'orange', 'yellow']
    n_adm = len(all_quat)

    global constantR
    global constantMobility
    global zeroMobility

    fig1 = plt.figure(figsize=(4, 4))
    ax = fig1.add_subplot(1, 1, 1)
    counter = 0
    for i_q, run_ID in enumerate(runIDs):

        # load data for this run
        folder = os.path.join(wd, 'Results', run_ID, 'original')
        files = os.listdir(folder)
        for i, f in enumerate(files):
            if f[0] == str(1) and f[-7:] == 'trn.csv':
                df_trn = pd.read_csv(os.path.join(folder, f))
                t_trn = df_trn['timestamp'].values
            elif f[0] == str(1) and f[-7:] == 'tst.csv':
                df_tst = pd.read_csv(os.path.join(folder, f))
                t_tst = df_tst['timestamp'].values
        t = list(np.arange(0, 150))

        folder = os.path.join(wd, 'Results', run_ID, 'parameters')
        files = os.listdir(folder)
        for i, f in enumerate(files):
            if f[-9:] == 'itted.pkl':
                infile = open(os.path.join(folder, f), 'rb')
                result = pickle.load(infile)
                infile.close()

            elif f[-9:] == 'fixed.pkl':
                infile = open(os.path.join(folder, f), 'rb')
                fixed_pars = pickle.load(infile)
                infile.close()

        # Get indices to use
        R_initU_save = []
        for j in range(0, len(result)):
            R_initU_save.append(result[j].x[5 * n_adm])
        df = pd.DataFrame(list(zip(R_initU_save)), columns=['Rstart'])
        inds_skip = list(
            np.where(df['Rstart'].values > np.mean(df['Rstart'].values) + 3 * np.std(df['Rstart'].values))[0])
        inds_skip = inds_skip + list(
            np.where(df['Rstart'].values < np.mean(df['Rstart'].values) - 3 * np.std(df['Rstart'].values))[0])

        n_tot = np.sum(fixed_pars[6])
        n_vaccinated = fixed_pars[6][vaccineGroup[i_q] - 1]
        f25 = fractions_vaccGroup[0] * n_tot / n_vaccinated
        fractions_vacc_group = [f25]
        fractions_vacc_all = [0, fractions_vaccGroup[0]]

        for ev in range(0, len(effVacsGroup)):
            effVac = effVacsGroup[ev]

            if counter == 0:
                # random distribution of vaccinated subjects
                for i_fv, frac_vac in enumerate(fractions_vacc_all):

                    if frac_vac == 0 and ev > 0:
                        continue

                    n_inf = predict_Vaccine(result[0].x, fixed_pars, 0, t, 1, effVac)

                    mean_ninf = np.zeros((n_inf.shape[0],))
                    min_ninf = np.zeros((n_inf.shape[0],))
                    max_ninf = np.zeros((n_inf.shape[0],))
                    for j in range(0, len(result)):
                        if np.isin(j, inds_skip):
                            continue

                        for q in all_quat:
                            if q == all_quat[0]:
                                mdl_cuminf_uncer = predict_Vaccine(result[j].x, fixed_pars, frac_vac, t, q, effVac)
                            else:
                                mdl_cuminf_uncer = mdl_cuminf_uncer + predict_Vaccine(result[j].x, fixed_pars, frac_vac,
                                                                                      t, q, effVac)

                        mean_ninf = mean_ninf + mdl_cuminf_uncer

                        if j > 0:
                            min_ninf = np.min(np.vstack((min_ninf, mdl_cuminf_uncer)), axis=0)
                            max_ninf = np.max(np.vstack((max_ninf, mdl_cuminf_uncer)), axis=0)
                        else:
                            min_ninf = mdl_cuminf_uncer
                            max_ninf = mdl_cuminf_uncer

                    # Average over boot straps
                    mean_ninf = mean_ninf / (len(result) - len(inds_skip))
                    if frac_vac == 0:

                        yV0_int = interp1d(t, mean_ninf)
                        cases = []
                        for it in range(0, len(t) - int(round(icu_stay))):
                            cases.append((yV0_int(it + icu_stay) - yV0_int(it)) * icu_perc)

                        yV0_int_cases = interp1d(100 * np.array(cases)[:cases.index(max(cases))] / icu_capa,
                                                 np.arange(cases.index(max(cases))))

                        xV0 = yV0_int_cases(50)
                        yV0_int = interp1d(t, mean_ninf)
                        yV0 = yV0_int(xV0)

                        # ax.plot(np.arange(len(cases)), 100*np.array(cases)/icu_capa, '-', linewidth=2,
                        # color=colors[counter], zorder=12,label='V0 - no vaccine')
                        ax.plot(t, mean_ninf, '-', linewidth=2, color=colors[counter], zorder=12,
                                label='V0 - no vaccine')
                    else:

                        mean_ninf1 = mean_ninf.copy()
                        yV1_int = interp1d(t, mean_ninf)
                        cases = []
                        for it in range(0, len(t) - int(round(icu_stay))):
                            cases.append((yV1_int(it + icu_stay) - yV1_int(it)) * icu_perc)

                        yV1_int_cases = interp1d(100 * np.array(cases)[:cases.index(max(cases))] / icu_capa,
                                                 np.arange(cases.index(max(cases))))

                        xV1 = yV1_int_cases(50)
                        yV1_int = interp1d(t, mean_ninf)
                        yV1 = yV1_int(xV1)

                        # ax.plot(np.arange(len(cases)), 100*np.array(cases)/icu_capa, '-', linewidth=2, color=colors[counter], zorder=12,
                        #         label='V1 - ' + f"{frac_vac * 100:.0f}" + '% randomly')  # with '+ f"{effVac*100:.0f}"+'% efficacy')
                        ax.plot(t, mean_ninf, '-', linewidth=2, color=colors[counter], zorder=12,
                                label='V1 - ' + f"{frac_vac * 100:.0f}" + '% randomly')

                    cases_min = []
                    cases_max = []
                    yV0_int_min = interp1d(t, min_ninf)
                    yV0_int_max = interp1d(t, max_ninf)
                    for it in range(0, len(t) - int(round(icu_stay))):
                        cases_min.append((yV0_int_min(it + icu_stay) - yV0_int_min(it)) * icu_perc)
                        cases_max.append((yV0_int_max(it + icu_stay) - yV0_int_max(it)) * icu_perc)

                    # ax.plot(np.arange(len(cases_min)), 100*np.array(cases_min)/icu_capa, '--',
                    #         color=colors[counter], zorder=1,alpha = 0.5, linewidth=1)
                    # ax.plot(np.arange(len(cases_min)), 100 * np.array(cases_max) / icu_capa, '--',
                    #         color=colors[counter], zorder=1, alpha=0.5,linewidth=1)
                    # ax.fill_between(np.arange(len(cases_min)), 100 * np.array(cases_min) / icu_capa,
                    #                 100 * np.array(cases_max) / icu_capa, color=colors[counter], zorder=3, alpha=0.25,
                    #                 linewidth=0)
                    ax.fill_between(t, min_ninf, max_ninf, color=colors[counter], zorder=3, alpha=0.25, linewidth=0)

                    counter += 1

            # Vaccine per group
            for i_fv, frac_vac in enumerate(fractions_vacc_group):

                if frac_vac == 0 and ev > 0:
                    continue

                mean_ninf = np.zeros((n_inf.shape[0],))
                min_ninf = np.zeros((n_inf.shape[0],))
                max_ninf = np.zeros((n_inf.shape[0],))
                for j in range(0, len(result)):
                    if np.isin(j, inds_skip):
                        continue

                    for q in all_quat:
                        if q == all_quat[0]:
                            mdl_cuminf_uncer = predict_VaccineOneGroup(result[j].x, fixed_pars, frac_vac, t, q,
                                                                       vaccineGroup[i_q], effVac)
                        else:
                            mdl_cuminf_uncer = mdl_cuminf_uncer + predict_VaccineOneGroup(result[j].x, fixed_pars,
                                                                                          frac_vac, t, q,
                                                                                          vaccineGroup[i_q], effVac)

                    mean_ninf = mean_ninf + mdl_cuminf_uncer

                    if j > 0:
                        min_ninf = np.min(np.vstack((min_ninf, mdl_cuminf_uncer)), axis=0)
                        max_ninf = np.max(np.vstack((max_ninf, mdl_cuminf_uncer)), axis=0)
                    else:
                        min_ninf = mdl_cuminf_uncer
                        max_ninf = mdl_cuminf_uncer

                # Average over boot straps
                mean_ninf = mean_ninf / (len(result) - len(inds_skip))

                yV2_int = interp1d(t, mean_ninf)
                cases = []
                for it in range(0, len(t) - int(round(icu_stay))):
                    cases.append((yV2_int(it + icu_stay) - yV2_int(it)) * icu_perc2)

                yV2_int_cases = interp1d(100 * np.array(cases)[:cases.index(max(cases))] / icu_capa,
                                         np.arange(cases.index(max(cases))))

                xV2 = yV2_int_cases(50)
                yV2_int = interp1d(t, mean_ninf)
                yV2 = yV2_int(xV2)

                if frac_vac == 0:
                    ax.plot(np.arange(len(cases)), 100 * np.array(cases) / icu_capa, '-', linewidth=2,
                            color=colors[counter], zorder=12, label='V0 - no vaccine')
                    ax.plot(t, mean_ninf, '-', linewidth=2, color=colors[counter], zorder=12,
                            label='V0 - no vaccine')
                else:
                    if i_fv == 1:
                        if vaccinateALLsenior:
                            ax.plot(t, mean_ninf, '-', linewidth=2, color='red', zorder=12,
                                    label='V3 - ' + f"{frac_vac * 100:.0f}" + '% T' + str(vaccineGroup[i_q]) + ' ' +
                                          names[i_q] + '=' + f"{fractions_vaccGroup[0] * 100:.0f}" + '% total')
                            # ax.plot(np.arange(len(cases)), 100 * np.array(cases) / icu_capa, '-', linewidth=2,
                            #         color=colors[counter], zorder=12,
                            #         label='V3 - ' + f"{frac_vac * 100:.0f}" + '% T' + str(vaccineGroup[i_q]) + ' ' +
                            #               names[i_q] + '=' + f"{fractions_vaccGroup[0] * 100:.0f}" + '% total')


                        else:
                            ax.plot(t, mean_ninf, '-', linewidth=2, color='red', zorder=12,
                                    label='V2 - ' + f"{frac_vac * 100:.0f}" + '% T' + str(vaccineGroup[i_q]) + ' ' +
                                          names[i_q] + '=' + f"{fractions_vaccGroup[0] * 100:.0f}" + '% total')
                            # ax.plot(np.arange(len(cases)), 100*np.array(cases)/icu_capa, '-', linewidth=2, color=colors[counter], zorder=12,
                            #         label='V2 - '+f"{frac_vac * 100:.0f}" + '% T'+ str(vaccineGroup[i_q]) +' '+ names[i_q] + '=' + f"{fractions_vaccGroup[0]*100:.0f}" + '% total')

                    else:
                        ax.plot(t, mean_ninf, '-', linewidth=2, color='red', zorder=12,
                                label='V2 - ' + f"{fractions_vaccGroup[0] * 100:.0f}" + '% from T' + str(
                                    vaccineGroup[i_q]) + "\n" + names[i_q])
                        # ax.plot(np.arange(len(cases)), 100*np.array(cases)/icu_capa, '-', linewidth=2, color='red', zorder=12,
                        #         label='V2 - '+ f"{fractions_vaccGroup[0]*100:.0f}"+ '% from T'+ str(vaccineGroup[i_q])+ "\n"+names[i_q])

                # Case differences
                delta = (mean_ninf1 / mean_ninf)
                list(mean_ninf1 / mean_ninf).index(np.nanmax(mean_ninf1 / mean_ninf))

                cases_min = []
                cases_max = []
                yV0_int_min = interp1d(t, min_ninf)
                yV0_int_max = interp1d(t, max_ninf)
                cases = []
                for it in range(0, len(t) - int(round(icu_stay))):
                    cases_min.append((yV0_int_min(it + icu_stay) - yV0_int_min(it)) * icu_perc2)
                    cases_max.append((yV0_int_max(it + icu_stay) - yV0_int_max(it)) * icu_perc2)
                # ax.fill_between(np.arange(len(cases_min)), 100*np.array(cases_min)/icu_capa,
                #                 100*np.array(cases_max)/icu_capa, color='red', zorder=3,alpha = 0.25, linewidth=0)
                # ax.plot(np.arange(len(cases_min)), 100 * np.array(cases_min) / icu_capa, '--',
                #         color='red', zorder=1, alpha=0.5, linewidth=1)
                # ax.plot(np.arange(len(cases_min)), 100 * np.array(cases_max) / icu_capa, '--',
                #         color='red', zorder=1, alpha=0.5, linewidth=1)
                ax.fill_between(t, min_ninf, max_ninf, color='red', zorder=3, alpha=0.25, linewidth=0)

                counter += 1

    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.xlabel('Time [days]', fontsize=14)
    plt.legend(prop={'size': 10}, frameon=True, facecolor='lightgrey', loc='lower right')
    plt.ylim([1, 300000])
    plt.xlim([0, len(cases_min)])
    plt.yscale('log')
    plt.ylabel('Number of Cases', fontsize=14)
    ax.patch.set_facecolor('lightgrey')
    plt.grid(color='white')
    # plt.hlines(y=50, xmin = 0, xmax =len(cases_min), linestyles = 'solid', color = 'darkorange')
    plt.scatter(xV0, yV0, color='black', marker='o', s=50, zorder=20)
    plt.scatter(xV1, yV1, color='darkblue', marker='o', s=50, zorder=20)
    plt.scatter(xV2, yV2, color='red', marker='o', s=50, zorder=20)
    plt.tight_layout()

    os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
    fig1.savefig('vaccineoneGroup_T' + str(vaccineGroup[i_q]) + '_mixed_all.png', dpi=250)
    fig1.savefig('vaccineoneGroup_T' + str(vaccineGroup[i_q]) + '_mixed_all.pdf', format='pdf')
    os.chdir(wd)

    print(xV0)
    print(xV1)
    print(xV2)
    print(yV0)
    print(yV1)
    print(yV2)

    return 0

def plot_results(run_ID):
    # run_ID = '3tiles_SENIOR_ANT_s1_unc50_Per'
    print(run_ID)

    # main_evalEffR(run_ID)

    # general setup
    n_bootstraps = 50
    saveParameters_tocsv = True
    plotConst = True
    plotfitR = False
    plotFitResult = False
    plotMobility = True
    plotHist = False
    plotVaccineOneGroup = False
    plotVaccineAll = True

    if run_ID == 'Final_uncert_20201204_1PHouseholds':
        vaccineGroup = 2
    else:
        vaccineGroup = 1

    mode = 'a'  # s: single, a: coupled
    start = dt.date(2020, 2, 22)  # start date of analysis (CHE:2/23,DEU:1/28)
    end = dt.date(2020, 4, 22)  # end date of analysis
    all_quat = [1, 2, 3]  # the summarized quaters of Basel to be analysed: choose from 1-9,'all'

    colors = ['orange', 'firebrick', 'gold', 'royalblue']
    colorsMean = ['black', 'darkblue', 'royalblue', 'darkred', 'crimson']
    fractions_vacc = [0, 0.3333333, 0.6666666]
    effVacs = [0.9, 0.7]
    fractions_vaccGroup = [0, 0.3333333]
    effVacsGroup = [0.9]

    global constantR
    global constantMobility
    global zeroMobility

    dates = np.array(pd.date_range(start, end))
    ts = pd.to_datetime(dates)
    dates = ts.strftime('%d %b').values
    if mode == 's':
        n_adm = 1
    else:
        n_adm = len(all_quat)

    # load data for this run
    folder = os.path.join(wd, 'Results', run_ID, 'parameters')
    files = os.listdir(folder)
    for i, f in enumerate(files):
        if f[-9:] == 'itted.pkl':
            infile = open(os.path.join(folder, f), 'rb')
            result = pickle.load(infile)
            infile.close()
            result0 = result[0]
            pars_all = result0.x

        elif f[-9:] == 'fixed.pkl':
            infile = open(os.path.join(folder, f), 'rb')
            fixed_pars = pickle.load(infile)
            infile.close()
            alpha_fix = fixed_pars[8]
            Adj = fixed_pars[3]

    folder = os.path.join(wd, 'Results', run_ID, 'original')
    files = os.listdir(folder)
    folderfit = os.path.join(wd, 'Results', run_ID, 'fitting')
    filesfit = os.listdir(folderfit)
    n_cmp = fixed_pars[1]

    if saveParameters_tocsv:
        for q in all_quat:
            # Save all parameters to csv
            normalizationFlst = []
            a_save = []
            b_save = []
            Tinf_save = []
            TinfUi_save = []
            R_initU_save = []
            Rabs = []
            Reff = []
            for j in range(0, n_bootstraps):
                a_save.append(result[j].x[8 * n_adm + 1 + q - 1])
                b_save.append(result[j].x[7 * n_adm + 1 + q - 1])
                Tinf_save.append(result[j].x[9 * n_adm + 1])
                TinfUi_save.append(result[j].x[10 * n_adm + 4])
                R_initU_save.append(result[j].x[5 * n_adm + q - 1])

                normFactor = np.sum(np.array(Adj)[q - 1, :])
                normalizationFlst.append(normFactor)
                if q > 1:
                    Rabs.append(result[j].x[5 * n_adm + q - 1] * result[j].x[5 * n_adm])
                    Reff.append(result[j].x[5 * n_adm + q - 1] * result[j].x[5 * n_adm] * normFactor)
                else:
                    Rabs.append(result[j].x[5 * n_adm + q - 1])
                    Reff.append(result[j].x[5 * n_adm + q - 1] * normFactor)

            df = pd.DataFrame(
                list(zip(a_save, b_save, Tinf_save, TinfUi_save, R_initU_save, Rabs, Reff, normalizationFlst)),
                columns=['a', 'b', 'Tinf', 'TinfUi', 'Rstart_rel', 'Rabs', 'Reff', 'normalizationF'])
            df.to_csv(os.path.join(wd, 'Results', run_ID, 'parameters', str(q) + '_parameters.csv'))

            # Get indices to use
            if q == all_quat[0]:
                inds_skip = list(
                    np.where(df['Rabs'].values > np.mean(df['Rabs'].values) + 3 * np.std(df['Rabs'].values))[0])
                inds_skip = inds_skip + list(
                    np.where(df['Rabs'].values < np.mean(df['Rabs'].values) - 3 * np.std(df['Rabs'].values))[0])

    all_RU_i_eff = []
    all_RU_i_abs = []

    normalizationF = np.zeros((len(all_quat), 1))
    for q in all_quat:

        # Get data
        for i, f in enumerate(files):
            if f[0] == str(q) and f[-7:] == 'trn.csv':
                df_trn = pd.read_csv(os.path.join(folder, f))
                t_trn = df_trn['timestamp'].values
                real_cuminf_trn = df_trn[str(q) + '_n_confirmed'].values
            elif f[0] == str(q) and f[-7:] == 'tst.csv':
                df_tst = pd.read_csv(os.path.join(folder, f))
                t_tst = df_tst['timestamp'].values
                real_cuminf_tst = df_tst[str(q) + '_n_confirmed'].values

        for i, f in enumerate(filesfit):
            if f[:7] == str(q) + '_n_asy':
                df = pd.read_csv(os.path.join(folderfit, f))
                n_asy = df[str(q) + '_n_asyminfected']
            elif f[:7] == str(q) + '_n_exp':
                df = pd.read_csv(os.path.join(folderfit, f))
                n_exp = df[str(q) + '_n_exposed']
            elif f[:7] == str(q) + '_n_inf':
                df = pd.read_csv(os.path.join(folderfit, f))
                n_inf = df[str(q) + '_n_infected']
            elif f[:7] == str(q) + '_n_sus':
                df = pd.read_csv(os.path.join(folderfit, f))
                n_sus = df[str(q) + '_n_susceptible']

        t = list(t_trn) + list(t_tst)
        ticks = np.arange(min(t_trn) + 2, max(t), 7)

        if useGradient:
            real_cuminf_trn = np.gradient(real_cuminf_trn)
            n_inf = np.gradient(n_inf)

        # First plot fit results
        if plotFitResult:
            print('Plotting fit!')

            for i in range(1):
                fig1 = plt.figure(figsize=(4, 4))
                ax = fig1.add_subplot(1, 1, 1)

                maxN = max(n_inf)
                r2_inf_trn = [r2_score(real_cuminf_trn, n_inf[:len(t_trn)])]
                print('\nR2 infected train: %.3f' % (r2_inf_trn[0]))

                if len(result) > 1:
                    mean_ninf = np.zeros(n_inf.shape)
                    mean_nUinfcum = np.zeros(n_inf.shape)
                    mean_nU = np.zeros(n_inf.shape)
                    mean_nE = np.zeros(n_inf.shape)
                    t = list(t_trn) + list(t_tst)
                    counter = 0
                    for j in range(0, n_bootstraps):

                        if np.isin(j, inds_skip):
                            continue

                        fit = solution_SEUI(t, result[j].x, fixed_pars).T
                        # mdl_sus_uncer  = fit[0 * n_adm + q - 1]
                        mdl_exp_uncer = fit[1 * n_adm + q - 1]
                        mdl_U_uncer = fit[2 * n_adm + q - 1]
                        mdl_Ui_uncer = fit[4 * n_adm + q - 1]
                        mdl_Ur_uncer = fit[5 * n_adm + q - 1]
                        mdl_inf_uncer = fit[3 * n_adm + q - 1]
                        mdl_cuminf_uncer = mdl_inf_uncer
                        mdl_cumUinf_uncer = mdl_Ui_uncer + mdl_Ur_uncer

                        if useGradient:
                            mdl_cumUinf_uncer = np.gradient(mdl_Ui_uncer + mdl_Ur_uncer)
                            mdl_cuminf_uncer = np.gradient(mdl_cuminf_uncer)

                        thisr2 = r2_score(real_cuminf_trn, mdl_cuminf_uncer[:len(t_trn)])
                        if j == 0:
                            thisRMSE = np.sqrt(np.mean((real_cuminf_trn - mdl_cuminf_uncer[:len(t_trn)]) ** 2))
                            print(thisRMSE)

                        r2_inf_trn.append(thisr2)

                        # if counter == 0:
                        #     ax.plot(t, mdl_exp_uncer,    '--', linewidth=0.5, color='red',  zorder=1)
                        #     ax.plot(t, mdl_U_uncer,      '--', linewidth=0.5, color='green',zorder=1)
                        #     ax.plot(t, mdl_cuminf_uncer, '--', linewidth=0.5, color='grey', zorder=1, label='Bootstraps')
                        #     ax.plot(t, mdl_cumUinf_uncer,'--', linewidth=0.5, color='blue', zorder=1)
                        # else:
                        #     ax.plot(t, mdl_cuminf_uncer, '--', linewidth=0.5, color='grey', zorder=1)
                        #     ax.plot(t, mdl_cumUinf_uncer,'--', linewidth=0.5, color='blue', zorder=1)

                        mean_ninf = mean_ninf + mdl_cuminf_uncer
                        mean_nUinfcum = mean_nUinfcum + mdl_cumUinf_uncer
                        mean_nU = mean_nU + mdl_U_uncer
                        mean_nE = mean_nE + mdl_exp_uncer

                        if j > 0:
                            min_nE = np.min(np.vstack((min_nE, mdl_exp_uncer)), axis=0)
                            max_nE = np.max(np.vstack((max_nE, mdl_exp_uncer)), axis=0)

                            min_nU = np.min(np.vstack((min_nU, mdl_U_uncer)), axis=0)
                            max_nU = np.max(np.vstack((max_nU, mdl_U_uncer)), axis=0)

                            min_nI = np.min(np.vstack((min_nI, mdl_cuminf_uncer)), axis=0)
                            max_nI = np.max(np.vstack((max_nI, mdl_cuminf_uncer)), axis=0)

                            min_nUicum = np.min(np.vstack((min_nUicum, mdl_cumUinf_uncer)), axis=0)
                            max_nUicum = np.max(np.vstack((max_nUicum, mdl_cumUinf_uncer)), axis=0)
                        else:
                            min_nE = mdl_exp_uncer
                            max_nE = mdl_exp_uncer

                            min_nU = mdl_U_uncer
                            max_nU = mdl_U_uncer

                            min_nI = mdl_cuminf_uncer
                            max_nI = mdl_cuminf_uncer

                            min_nUicum = mdl_cumUinf_uncer
                            max_nUicum = mdl_cumUinf_uncer

                        counter = counter + 1
                        maxN = max(
                            [maxN, max(mdl_inf_uncer), max(mdl_cumUinf_uncer), max(mdl_U_uncer), max(mdl_exp_uncer)])

                    # Average
                    mean_nE = mean_nE / counter
                    mean_nU = mean_nU / counter
                    mean_ninf = mean_ninf / counter
                    mean_nUinfcum = mean_nUinfcum / counter

                    # Plot Average
                    if q == 1:
                        ax.plot(t, mean_nE, '-', linewidth=2, color='black', zorder=12, label='E')
                        ax.plot(t, mean_nU, '--', linewidth=2, color='darkred', zorder=12, label='P')
                        ax.plot(t, mean_ninf, ':', linewidth=2, color='firebrick', zorder=12,
                                label='I, RMSE = ' + f"{thisRMSE:.1f}")
                        ax.plot(t, mean_nUinfcum, '-', linewidth=2, color='lightcoral', zorder=12, label='$U_i$+$U_r$')
                        ax.fill_between(t, min_nE, max_nE, color='black', zorder=3, alpha=0.25, linewidth=0)
                        ax.fill_between(t, min_nU, max_nU, color='darkred', zorder=3, alpha=0.25, linewidth=0)
                        ax.fill_between(t, min_nI, max_nI, color='firebrick', zorder=3, alpha=0.25, linewidth=0)
                        ax.fill_between(t, min_nUicum, max_nUicum, color='lightcoral', zorder=3, alpha=0.25,
                                        linewidth=0)
                    elif q == 2:
                        ax.plot(t, mean_nE, '-', linewidth=2, color='black', zorder=12, label='E')
                        ax.plot(t, mean_nU, '--', linewidth=2, color='darkorange', zorder=12, label='P')
                        ax.plot(t, mean_ninf, ':', linewidth=2, color='gold', zorder=12,
                                label='I, RMSE = ' + f"{thisRMSE:.1f}")
                        ax.plot(t, mean_nUinfcum, '-', linewidth=2, color='yellow', zorder=12, label='$U_i$+$U_r$')
                        ax.fill_between(t, min_nE, max_nE, color='black', zorder=3, alpha=0.25, linewidth=0)
                        ax.fill_between(t, min_nU, max_nU, color='darkorange', zorder=3, alpha=0.25, linewidth=0)
                        ax.fill_between(t, min_nI, max_nI, color='gold', zorder=3, alpha=0.25, linewidth=0)
                        ax.fill_between(t, min_nUicum, max_nUicum, color='yellow', zorder=3, alpha=0.25, linewidth=0)
                    else:
                        ax.plot(t, mean_nE, '-', linewidth=2, color='black', zorder=12, label='E')
                        ax.plot(t, mean_nU, '--', linewidth=2, color='darkblue', zorder=12, label='P')
                        ax.plot(t, mean_ninf, ':', linewidth=2, color='royalblue', zorder=12,
                                label='I, RMSE = ' + f"{thisRMSE:.1f}")
                        ax.plot(t, mean_nUinfcum, '-', linewidth=2, color='cornflowerblue', zorder=12,
                                label='$U_i$+$U_r$')
                        ax.fill_between(t, min_nE, max_nE, color='black', zorder=3, alpha=0.25, linewidth=0)
                        ax.fill_between(t, min_nU, max_nU, color='darkblue', zorder=3, alpha=0.25, linewidth=0)
                        ax.fill_between(t, min_nI, max_nI, color='royalblue', zorder=3, alpha=0.25, linewidth=0)
                        ax.fill_between(t, min_nUicum, max_nUicum, color='cornflowerblue', zorder=3, alpha=0.25,
                                        linewidth=0)

                # Format plot
                ax.set_xticks(ticks)
                plt.yticks(fontsize=16)
                int_ticks = [int(i) for i in ticks]
                ax.set_xticklabels(dates[int_ticks], rotation=45, fontsize=16, ha='right')
                # plt.xlabel('Date',fontsize=14)
                plt.grid()
                plt.xlim([0, len(n_inf) - 1])
                if i == 1:
                    ax.scatter(t_trn, real_cuminf_trn, color='black', marker='o', s=20, label='Data T' + str(q),
                               zorder=10)
                    plt.legend(prop={'size': 10}, frameon=False, loc="upper left")
                    plt.yticks(fontsize=10)
                    plt.xticks(fontsize=10)
                    plt.ylim([1, 11000])  # 1.2*maxN])
                    plt.xlim([0, len(t) - 1])
                    plt.yscale('log')
                    plt.ylabel('Number of Cases', fontsize=14)
                    plt.tight_layout()
                    ax.patch.set_facecolor('lightgrey')
                    plt.grid(color='white')
                    plt.tight_layout()

                    core_name = f[12:-4]
                    os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
                    fig1.savefig(str(q) + '_infTime_lin.png', dpi=250)
                    fig1.savefig(str(q) + '_infTime_lin.pdf', format='pdf')
                    os.chdir(wd)

                elif i == 0:
                    ax.scatter(t_trn, real_cuminf_trn, color='black', marker='o', s=10, label='Data T' + str(q),
                               zorder=15)
                    plt.legend(prop={'size': 10}, frameon=False, loc="upper left")

                    plt.yscale('log')
                    plt.yticks(fontsize=10)
                    plt.xticks(fontsize=10)
                    plt.ylim([1, 11000])  # 1.2*maxN])
                    plt.xlim([0, len(t) - 1])
                    plt.yscale('log')
                    plt.ylabel('Number of Cases', fontsize=14)
                    plt.tight_layout()
                    ax.patch.set_facecolor('lightgrey')
                    plt.grid(color='white')
                    plt.tight_layout()

                    core_name = f[12:-4]
                    os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
                    fig1.savefig(str(q) + '_infTime_log.png', dpi=250)
                    fig1.savefig(str(q) + '_infTime_log.pdf', format='pdf')
                    os.chdir(wd)

        # Reproductive number
        if plotfitR:
            print('Plotting R!')
            time_dep = fixed_pars[11]
            idx = q - 1

            a = result[0].x[2 * n_adm + 1 + n_adm + (n_cmp - 1) * n_adm:3 * n_adm + 1 + n_adm + (n_cmp - 1) * n_adm]
            b = result[0].x[2 * n_adm + 1 + n_adm + (n_cmp - 1) * n_adm:3 * n_adm + 1 + n_adm + (n_cmp - 1) * n_adm]
            Tinf = result[0].x[(n_cmp - 1) * n_adm + 4 * n_adm + 1]
            TinfUi = result[0].x[(n_cmp - 1) * n_adm + 5 * n_adm + 4]
            R_initU = result[0].x[(n_cmp - 1) * n_adm:n_adm + (n_cmp - 1) * n_adm]
            R_redU_frac = result[0].x[(n_cmp - 1) * n_adm + n_adm + 1:(n_adm + 1) + n_adm + (n_cmp - 1) * n_adm]

            if useMultiplicModel:
                factor = np.asarray(Adj)[idx, :].sum()

                if useStretchSocial:
                    stretch = 0.19
                else:
                    stretch = 0
                if useRelativeR:
                    if idx == 0:
                        RU_abs = [factor * stretchFun(t, 51, stretch) * R_initU[idx] for t in t_trn]
                    else:
                        RU_abs = [factor * R_initU[0] * stretchFun(t, 51, stretch) * R_initU[idx] for t in t_trn]

                else:
                    RU_abs = [factor * stretchFun(t, 51, stretch) * R_initU[idx] for t in t_trn]
            else:
                factor = 1
                if useSigmoid:
                    if useRelativeR:
                        if idx == 0:
                            RU_abs = [sigmoid_R(R_initU[idx], R_redU_frac[idx], a[0], b[0], t) for t in t_trn]
                        else:
                            RU_abs = [R_initU[idx] * sigmoid_R(R_initU[0], R_redU_frac[0], a[0], b[0], t) for t in
                                      t_trn]

                    else:
                        RU_abs = [sigmoid_R(R_initU[idx], R_redU_frac[idx], a[0], b[0], t) for t in t_trn]
                else:
                    if useRelativeR:
                        if idx == 0:
                            RU_abs = [R_initU[idx] * time_dep_soc(t) for t in
                                      t_trn]  # [timeDep_R(R_initU[idx], R_redU_frac[idx], time_dep(t)) for t in t_trn]
                        else:
                            RU_abs = [R_initU[idx] * R_initU[0] * time_dep_soc(t) for t in
                                      t_trn]  # [R_initU[idx]*timeDep_R(R_initU[0], R_redU_frac[0], time_dep(t)) for t in t_trn]

                    else:
                        RU_abs = [R_initU[idx] * time_dep_soc(t) for t in
                                  t_trn]  # RU_abs = [timeDep_R(R_initU[idx], R_redU_frac[idx], time_dep(t)) for t in t_trn]

            # Plot
            fig2 = plt.figure(figsize=(4, 4))
            ax2 = fig2.add_subplot(1, 1, 1)
            # ax2.plot(t_trn, RU_abs,color = 'blue', label='Reproductive number before symptoms', zorder=15)

            all_a_uncert = [a]
            all_b_uncert = [b]

            if useRelativeR:
                if idx == 0:
                    all_R_initU_uncert_abs = [factor * R_initU[idx]]
                    all_R_redU_frac_uncert_abs = [R_redU_frac[idx]]
                else:
                    all_R_initU_uncert_abs = [factor * R_initU[idx] * R_initU[0]]
                    all_R_redU_frac_uncert_abs = [R_redU_frac[idx]]
            else:
                all_R_initU_uncert_abs = [factor * R_initU[idx]]
                all_R_redU_frac_uncert_abs = [R_redU_frac[idx]]

            # Bootstrap
            allRU_abs = np.zeros(np.array(RU_abs).shape)
            allRUmax_abs = max(RU_abs)
            if len(result) > 1:
                counter_bootstraps = 0
                for j in range(0, n_bootstraps):

                    if np.isin(j, inds_skip):
                        continue
                    else:
                        counter_bootstraps += 1

                    Tinf_uncer = result[j].x[9 * n_adm + 1]
                    a_uncer = result[j].x[8 * n_adm + 1:9 * n_adm + 1]
                    b_uncer = result[j].x[7 * n_adm + 1:8 * n_adm + 1]
                    R_initU_uncer = result[j].x[5 * n_adm:6 * n_adm]
                    R_redU_frac_uncer = result[j].x[6 * n_adm + 1:7 * n_adm + 1]
                    all_a_uncert.append(a_uncer)
                    all_b_uncert.append(b_uncer)
                    if useRelativeR:
                        if idx == 0:
                            all_R_initU_uncert_abs.append(factor * R_initU_uncer[idx])
                        else:
                            all_R_initU_uncert_abs.append(factor * R_initU_uncer[idx] * R_initU_uncer[0])
                    else:
                        all_R_initU_uncert_abs.append(factor * R_initU_uncer[idx])
                    all_R_redU_frac_uncert_abs.append(R_redU_frac_uncer[idx])

                    if useMultiplicModel:

                        if useStretchSocial:
                            stretch = 0.19
                        else:
                            stretch = 0
                        if useRelativeR:
                            if idx == 0:
                                RU_uncer_abs = [factor * stretchFun(t, 51, stretch) * R_initU_uncer[idx] for t in t_trn]
                            else:
                                RU_uncer_abs = [
                                    factor * R_initU_uncer[0] * stretchFun(t, 51, stretch) * R_initU_uncer[idx] for t in
                                    t_trn]

                        else:
                            RU_uncer_abs = [factor * stretchFun(t, 51, stretch) * R_initU_uncer[idx] for t in t_trn]
                    else:

                        if useSigmoid:

                            if useRelativeR:
                                if idx == 0:
                                    RU_uncer_abs = [
                                        sigmoid_R(R_initU_uncer[idx], R_redU_frac_uncer[idx], a_uncer[0], b_uncer[0],
                                                  t) for t in t_trn]
                                else:
                                    RU_uncer_abs = [
                                        R_initU_uncer[idx] * sigmoid_R(R_initU_uncer[0], R_redU_frac_uncer[0],
                                                                       a_uncer[0], b_uncer[0],
                                                                       t) for t in t_trn]
                            else:
                                RU_uncer_abs = [
                                    sigmoid_R(R_initU_uncer[idx], R_redU_frac_uncer[idx], a_uncer[0], b_uncer[0],
                                              t) for t in t_trn]
                        else:

                            if useRelativeR:
                                if idx == 0:
                                    RU_uncer_abs = [R_initU_uncer[idx] * time_dep_soc(t) for t in
                                                    t_trn]  # [timeDep_R(R_initU_uncer[idx], R_redU_frac_uncer[idx], time_dep(t)) for t in t_trn]
                                else:
                                    RU_uncer_abs = [R_initU_uncer[idx] * R_initU_uncer[0] * time_dep_soc(t) for t in
                                                    t_trn]  # [R_initU_uncer[idx]*timeDep_R(R_initU_uncer[0], R_redU_frac_uncer[0], time_dep(t))
                                    # for t in t_trn]
                            else:
                                RU_uncer_abs = [R_initU_uncer[idx] * time_dep_soc(t) for t in
                                                t_trn]  # [timeDep_R(R_initU_uncer[idx], R_redU_frac_uncer[idx], time_dep(t)) for t in t_trn]

                    # if j == 0:
                    #     ax2.plot(t_trn, RU_uncer_abs, '--', linewidth=0.5, color='grey', zorder=0.1 * j,label = 'Bootstraps')
                    # else:
                    #     ax2.plot(t_trn, RU_uncer_abs, '--', linewidth=0.5, color='grey', zorder=0.1 * j)

                    if j > 0:
                        min_RU_uncer_abs = np.min(np.vstack((min_RU_uncer_abs, RU_uncer_abs)), axis=0)
                        max_RU_uncer_abs = np.max(np.vstack((max_RU_uncer_abs, RU_uncer_abs)), axis=0)
                    else:
                        min_RU_uncer_abs = RU_uncer_abs
                        max_RU_uncer_abs = RU_uncer_abs

                    allRU_abs = allRU_abs + RU_uncer_abs
                    allRUmax_abs = max([allRUmax_abs, max(RU_uncer_abs)])

                # Mean of all runs
                allRU_mean_abs = allRU_abs / counter_bootstraps
                ax2.plot(t_trn, allRU_mean_abs, '-', linewidth=2, color=colors[q], zorder=10, label='T' + str(q))
                ax2.fill_between(t, min_RU_uncer_abs, max_RU_uncer_abs, color=colors[q], zorder=3, alpha=0.25,
                                 linewidth=0)

            ax2.set_xticks(ticks)
            int_ticks = [int(i) for i in ticks]
            ax2.set_xticklabels(dates[int_ticks], rotation=45, ha='right')
            # plt.xlabel('Time [days]',fontsize = 16)
            plt.ylabel('Reproductive Number', fontsize=14)
            plt.ylim([0, 3.5])  # 1.2 * allRUmax_abs])
            plt.xlim([0, len(t)])  # 1.2 * allRUmax_abs])
            plt.legend(prop={'size': 14}, loc="upper right", frameon=False)
            plt.yticks(fontsize=10)
            plt.xticks(fontsize=10)
            ax2.patch.set_facecolor('lightgrey')
            plt.grid(color='white')
            plt.tight_layout()

            os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
            fig2.savefig(str(q) + '_R_abs.png', dpi=250)
            fig2.savefig(str(q) + '_R_abs.pdf', format='pdf')
            all_RU_i_abs.append(all_R_initU_uncert_abs)

    # Mobility scenarios
    if plotMobility:

        print('Plotting monbility!')

        for i in range(2):
            fig1 = plt.figure(figsize=(4, 4))
            ax = fig1.add_subplot(1, 1, 1)
            maxN = max(n_inf)
            maxN_const = maxN
            r2_inf_trn = [r2_score(real_cuminf_trn, n_inf[:len(t_trn)])]
            print('\nR2 infected train: %.3f' % (r2_inf_trn[0]))

            if plotConst:
                constantMobility = True
                fit_const = solution_SEUI(t, result[0].x, fixed_pars).T
                zeroMobility = True
                fit_zero = solution_SEUI(t, result[0].x, fixed_pars).T
                zeroMobility = False
                constantMobility = False

            if len(result) > 1:
                t = list(t_trn) + list(t_tst)
                mean_ninf = np.zeros(n_inf.shape)
                mean_ninf_const = np.zeros(n_inf.shape)
                mean_ninf_zero = np.zeros(n_inf.shape)
                counter = 0
                indices = []
                for j in range(0, n_bootstraps):
                    if np.isin(j, inds_skip):
                        continue
                    # print('At step ' + str(j) + ' of ' + str(len(result)))
                    zeroMobility = False
                    constantMobility = False
                    fit = solution_SEUI(t, result[j].x, fixed_pars).T
                    constantMobility = True
                    fit_const = solution_SEUI(t, result[j].x, fixed_pars).T
                    zeroMobility = True
                    fit_zero = solution_SEUI(t, result[j].x, fixed_pars).T
                    zeroMobility = False
                    constantMobility = False
                    for q in all_quat:

                        if q == all_quat[0]:
                            mdl_ui_r_uncer = fit[4 * n_adm + q - 1] + fit[5 * n_adm + q - 1]
                            mdl_cuminf_uncer = fit[3 * n_adm + q - 1]
                            mdl_cuminf_uncer_const = fit_const[3 * n_adm + q - 1]
                            mdl_ui_r_uncer_const = fit_const[4 * n_adm + q - 1] + fit[5 * n_adm + q - 1]
                            mdl_cuminf_uncer_zero = fit_zero[3 * n_adm + q - 1]
                            mdl_ui_r_uncer_zero = fit_zero[4 * n_adm + q - 1] + fit[5 * n_adm + q - 1]
                        else:
                            mdl_ui_r_uncer = mdl_ui_r_uncer + fit[4 * n_adm + q - 1] + fit[5 * n_adm + q - 1]
                            mdl_cuminf_uncer = mdl_cuminf_uncer + fit[3 * n_adm + q - 1]
                            mdl_cuminf_uncer_const = mdl_cuminf_uncer_const + fit_const[3 * n_adm + q - 1]
                            mdl_ui_r_uncer_const = mdl_ui_r_uncer_const + fit_const[4 * n_adm + q - 1] + fit[
                                5 * n_adm + q - 1]
                            mdl_cuminf_uncer_zero = mdl_cuminf_uncer_zero + fit_zero[3 * n_adm + q - 1]
                            mdl_ui_r_uncer_zero = mdl_ui_r_uncer_zero + fit_zero[4 * n_adm + q - 1] + fit[
                                5 * n_adm + q - 1]

                    # if j == 1:
                    #     ax.plot(t, mdl_cuminf_uncer_const+mdl_ui_r_uncer_const, '--', linewidth=0.5, color='red', zorder=1)#label='Bootstraps const. Mobility')
                    #     ax.plot(t, mdl_cuminf_uncer_zero+mdl_ui_r_uncer_zero, '--', linewidth=0.5, color='green', zorder=1)
                    #     #ax.plot(t, mdl_cuminf_uncer+mdl_ui_r_uncer, '--', linewidth=0.5, color='grey', zorder=1,label='Bootstraps')
                    # else:
                    #     ax.plot(t, mdl_cuminf_uncer_const+mdl_ui_r_uncer_const, '--', linewidth=0.5, color='red', zorder=1)
                    #     ax.plot(t, mdl_cuminf_uncer_zero+mdl_ui_r_uncer_zero, '--', linewidth=0.5, color='green', zorder=1)
                    # ax.plot(t, mdl_cuminf_uncer+mdl_ui_r_uncer, '--', linewidth=0.5, color='grey', zorder=1)

                    if j > 0:
                        min_mob = np.min(np.vstack((min_mob, mdl_cuminf_uncer + mdl_ui_r_uncer)), axis=0)
                        max_mob = np.max(np.vstack((max_mob, mdl_cuminf_uncer + mdl_ui_r_uncer)), axis=0)
                        min_zero = np.min(np.vstack((min_zero, mdl_cuminf_uncer_zero + mdl_ui_r_uncer_zero)), axis=0)
                        max_zero = np.max(np.vstack((max_zero, mdl_cuminf_uncer_zero + mdl_ui_r_uncer_zero)), axis=0)
                        min_const = np.min(np.vstack((min_const, mdl_cuminf_uncer_const + mdl_ui_r_uncer_const)),
                                           axis=0)
                        max_const = np.max(np.vstack((max_const, mdl_cuminf_uncer_const + mdl_ui_r_uncer_const)),
                                           axis=0)
                    else:
                        min_mob = mdl_cuminf_uncer + mdl_ui_r_uncer
                        max_mob = mdl_cuminf_uncer + mdl_ui_r_uncer
                        min_zero = mdl_cuminf_uncer_zero + mdl_ui_r_uncer_zero
                        max_zero = mdl_cuminf_uncer_zero + mdl_ui_r_uncer_zero
                        min_const = mdl_cuminf_uncer_const + mdl_ui_r_uncer_const
                        max_const = mdl_cuminf_uncer_const + mdl_ui_r_uncer_const

                    mean_ninf_const = mean_ninf_const + mdl_cuminf_uncer_const + mdl_ui_r_uncer_const
                    maxN_const = max([maxN_const, max(mdl_cuminf_uncer_const + mdl_ui_r_uncer_const)])
                    mean_ninf_zero = mean_ninf_zero + mdl_cuminf_uncer_zero + mdl_ui_r_uncer_zero
                    maxN_const = max([maxN_const, max(mdl_cuminf_uncer_zero + mdl_ui_r_uncer_zero)])
                    maxN = max([maxN, maxN_const])
                    mean_ninf = mean_ninf + mdl_cuminf_uncer + mdl_ui_r_uncer
                    counter = counter + 1
                    maxN = max([maxN, max(mdl_cuminf_uncer + mdl_ui_r_uncer)])
                    indices.append(j)

                mean_ninf = mean_ninf / counter
                ax.plot(t, mean_ninf * 0.001, '-', linewidth=2, color='black', zorder=12, label='MO - observed')
                mean_ninf_const = mean_ninf_const / counter
                ax.fill_between(t, min_mob * 0.001, max_mob * 0.001, color='black', zorder=1, alpha=0.25, linewidth=0)

                ax.plot(t, mean_ninf_const * 0.001, '--', linewidth=2, color=colors[3], zorder=11,
                        label='M1 - full mobility')
                ax.fill_between(t, min_const * 0.001, max_const * 0.001, color=colors[3], zorder=1, alpha=0.25,
                                linewidth=0)

                mean_ninf_zero = mean_ninf_zero / counter
                ax.plot(t, mean_ninf_zero * 0.001, ':', linewidth=2, color=colors[1], zorder=11,
                        label='M2 - no mobility')
                ax.fill_between(t, min_zero * 0.001, max_zero * 0.001, color=colors[1], zorder=1, alpha=0.25,
                                linewidth=0)

            ax.set_xticks(ticks)
            plt.yticks(fontsize=10)
            int_ticks = [int(i) for i in ticks]
            ax.set_xticklabels(dates[int_ticks], rotation=45, fontsize=10, ha='right')
            # plt.xlabel(fontsize=16)
            ax.patch.set_facecolor('lightgrey')
            plt.grid(color='white')

            if i == 1:
                plt.yticks(fontsize=10)
                plt.ylim([0, 8])  # 1.2*maxN])#1.05 * max(max(real_cuminf_trn), max(real_cuminf_trn))])
                plt.legend(prop={'size': 10}, frameon=False, loc="upper left")
                plt.ylabel('Number of Cases [x1000]', fontsize=14)
                plt.tight_layout()
                plt.xlim([0, len(t) - 1])
                plt.tight_layout()
                core_name = f[12:-4]
                os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
                fig1.savefig('mobility_lin.png', dpi=250)
                fig1.savefig('mobility_lin.pdf', format='pdf')
                os.chdir(wd)
            elif i == 0:
                plt.yticks(fontsize=10)
                plt.legend(prop={'size': 10}, frameon=False, loc="upper left")
                plt.ylim([1, 8])
                plt.yscale('log')
                plt.ylabel('Number of Cases', fontsize=14)
                plt.xlim([0, len(t) - 1])
                plt.tight_layout()

                core_name = f[12:-4]
                os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
                fig1.savefig('mobility_log.png', dpi=250)
                fig1.savefig('mobility_log.pdf', format='pdf')
                os.chdir(wd)

    if plotVaccineOneGroup:
        print('Plotting vaccine scenario 2!')
        n_tot = np.sum(fixed_pars[6])
        n_vaccinated = fixed_pars[6][vaccineGroup - 1]
        f25 = 0.3333 * n_tot / n_vaccinated
        abs90 = 0.9 * n_vaccinated / n_tot

        names_vac_group = [0, 33]  # , abs90*100]
        fractions_vacc_group = [0, f25]

        fig1 = plt.figure(figsize=(6, 6))
        ax = fig1.add_subplot(1, 1, 1)
        counter = 0
        for ev in range(0, len(effVacsGroup)):
            effVac = effVacsGroup[ev]

            for i_fv, frac_vac in enumerate(fractions_vacc_group):

                if frac_vac == 0 and ev > 0:
                    continue

                n_inf = predict_Vaccine(result[0].x, fixed_pars, 0, t, q, effVac)

                mean_ninf = np.zeros((n_inf.shape[0],))
                min_ninf = np.zeros((n_inf.shape[0],))
                max_ninf = np.zeros((n_inf.shape[0],))
                for j in range(0, n_bootstraps):
                    if np.isin(j, inds_skip):
                        continue

                    for q in all_quat:
                        if q == all_quat[0]:
                            mdl_cuminf_uncer = predict_VaccineOneGroup(result[j].x, fixed_pars, frac_vac, t, q,
                                                                       vaccineGroup, effVac)
                        else:
                            mdl_cuminf_uncer = mdl_cuminf_uncer + predict_VaccineOneGroup(result[j].x, fixed_pars,
                                                                                          frac_vac, t, q, vaccineGroup,
                                                                                          effVac)

                    # if j == 1 and i_fv==0:
                    #     ax.plot(t, mdl_cuminf_uncer, '--', linewidth=0.5, color=colors[i_fv],zorder=1,label='Bootstraps ') #+str(frac_vac)+ '%')
                    # else:
                    #     ax.plot(t, mdl_cuminf_uncer, '--', linewidth=0.5, color=colors[i_fv], zorder=1)
                    mean_ninf = mean_ninf + mdl_cuminf_uncer

                    if j > 0:
                        min_ninf = np.min(np.vstack((min_ninf, mdl_cuminf_uncer)), axis=0)
                        max_ninf = np.max(np.vstack((max_ninf, mdl_cuminf_uncer)), axis=0)
                    else:
                        min_ninf = mdl_cuminf_uncer
                        max_ninf = mdl_cuminf_uncer

                # Average over boot straps
                mean_ninf = mean_ninf / (n_bootstraps - len(inds_skip))
                if frac_vac == 0:
                    ax.plot(t, mean_ninf, '-', linewidth=2, color=colorsMean[counter], zorder=12,
                            label='No vaccination')
                else:
                    if i_fv == 1:
                        ax.plot(t, mean_ninf, '-', linewidth=2, color=colorsMean[counter], zorder=12,
                                label=f"{frac_vac * 100:.0f}" + '% T' + str(
                                    vaccineGroup) + '=' + f"{names_vac_group[i_fv]:.0f}" + '% total,' + f"{effVac * 100:.0f}" + '% efficacy')

                    else:
                        ax.plot(t, mean_ninf, '--', linewidth=2, color=colorsMean[counter], zorder=12,
                                label=f"{frac_vac * 100:.0f}" + '% T' + str(
                                    vaccineGroup) + '=' + f"{names_vac_group[i_fv]:.0f}" + '% total,' + f"{effVac * 100:.0f}" + '% efficacy')

                ax.fill_between(t, min_ninf, max_ninf, color=colorsMean[counter], zorder=3, alpha=0.25, linewidth=0)
                counter += 1

        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        plt.xlabel('Time [days]', fontsize=16)
        plt.legend(prop={'size': 10}, frameon=False)
        plt.ylim([1, 200000])
        plt.xlim([0, len(t) - 1])
        plt.yscale('log')
        plt.ylabel('Number of Cases', fontsize=16)
        plt.tight_layout()
        ax.patch.set_facecolor('lightgrey')
        plt.grid(color='white')

        core_name = f[12:-4]
        os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
        fig1.savefig('vaccineoneGroup_sum_eff.png', dpi=250)
        fig1.savefig('vaccineoneGroup_sum_eff.pdf', format='pdf')
        os.chdir(wd)

    if plotVaccineAll:
        tvac = np.arange(0, 150)
        print('Plotting vaccine scenario 1!')

        fig1 = plt.figure(figsize=(4, 4))
        ax = fig1.add_subplot(1, 1, 1)
        counter = 0
        for ev in range(0, len(effVacs)):
            effVac = effVacs[ev]

            for i_fv, frac_vac in enumerate(fractions_vacc):

                if frac_vac == 0 and ev > 0:
                    continue

                n_inf = predict_Vaccine(result[0].x, fixed_pars, 0, tvac, q, effVac)

                mean_ninf = np.zeros((n_inf.shape[0],))
                min_ninf = np.zeros((n_inf.shape[0],))
                max_ninf = np.zeros((n_inf.shape[0],))
                for j in range(0, n_bootstraps):
                    if np.isin(j, inds_skip):
                        continue

                    for q in all_quat:
                        if q == all_quat[0]:
                            mdl_cuminf_uncer = predict_Vaccine(result[j].x, fixed_pars, frac_vac, tvac, q, effVac)
                        else:
                            mdl_cuminf_uncer = mdl_cuminf_uncer + predict_Vaccine(result[j].x, fixed_pars, frac_vac,
                                                                                  tvac, q, effVac)

                    # if j == 1 and i_fv==0:
                    #     ax.plot(t, mdl_cuminf_uncer, '--', linewidth=0.5, color=colors[i_fv],zorder=1,label='Bootstraps ') #+str(frac_vac)+ '%')
                    # else:
                    #     ax.plot(t, mdl_cuminf_uncer, '--', linewidth=0.5, color=colors[i_fv], zorder=1)
                    mean_ninf = mean_ninf + mdl_cuminf_uncer

                    if j > 0:
                        min_ninf = np.min(np.vstack((min_ninf, mdl_cuminf_uncer)), axis=0)
                        max_ninf = np.max(np.vstack((max_ninf, mdl_cuminf_uncer)), axis=0)
                    else:
                        min_ninf = mdl_cuminf_uncer
                        max_ninf = mdl_cuminf_uncer

                # Average over boot straps
                mean_ninf = mean_ninf / (n_bootstraps - len(inds_skip))
                if frac_vac == 0:
                    ax.plot(tvac, mean_ninf, '-', linewidth=2, color=colorsMean[counter], zorder=12, label='V0')
                    xV0_int = interp1d(mean_ninf, tvac)
                    xV0 = xV0_int(10000)
                    # plt.axvline(x=xV0, color=colorsMean[counter], ls=':')
                else:
                    if i_fv == 1:
                        ax.plot(tvac, mean_ninf, '-', linewidth=2, color=colorsMean[counter], zorder=12,
                                label='V1, ' + f"{frac_vac * 100:.0f}" + '% ' + ' ' + f"{effVac * 100:.0f}" + '% eff.')
                    else:
                        ax.plot(tvac, mean_ninf, '--', linewidth=2, color=colorsMean[counter], zorder=12,
                                label='V1, ' + f"{frac_vac * 100:.0f}" + '% ' + ' ' + f"{effVac * 100:.0f}" + '% eff.')

                try:
                    xV0_int = interp1d(mean_ninf, tvac)
                    yV0_int = interp1d(tvac, mean_ninf)

                    cases = []
                    for it in range(0, len(tvac) - 15):
                        cases.append((yV0_int(it + 14.5) - yV0_int(it)) * 0.01)

                    xV0 = np.where(np.array(cases) > 22)[0][0] + 14.5
                    yV0 = yV0_int(xV0)
                    plt.scatter(xV0, yV0, color=colorsMean[counter], marker='o', s=50, zorder=20)
                except:
                    print('no 10000')
                ax.fill_between(tvac, min_ninf, max_ninf, color=colorsMean[counter], zorder=1, alpha=0.25, linewidth=0)
                counter += 1

        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        plt.xlabel('Time [days]', fontsize=14)
        plt.legend(prop={'size': 10}, frameon=True, facecolor='lightgrey')
        plt.ylim([0.01, 200000])
        plt.xlim([0, len(tvac) - 1])
        plt.yscale('log')
        plt.ylabel('Number of Cases', fontsize=14)
        plt.tight_layout()
        ax.patch.set_facecolor('lightgrey')
        plt.grid(color='white')
        plt.tight_layout()

        core_name = f[12:-4]
        os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
        fig1.savefig('vaccineAll.png', dpi=250)
        fig1.savefig('vaccineAll.pdf', format='pdf')
        os.chdir(wd)

    # Histograms
    if plotHist:

        # print('Plotting effective histograms!')
        # fig = plt.figure(figsize=(6, 3))
        # ax1 = fig.add_subplot()
        #
        # minbin = 0.
        # maxbin = 3.5
        # step = (maxbin - minbin) / 20
        # if step > 0.02:
        #     step = 0.02
        # bins = np.arange(minbin, maxbin, step)
        #
        # for i in range(0, len(all_RU_i_eff)):
        #
        #     # Initial R
        #     R_eff = np.array(all_RU_i_eff[i])
        #
        #     # add histogram to plot
        #     ax1.hist(R_eff, bins=bins, lw=1, ec='w', alpha=.667, label='Q'+str(i + 1), density=False, color=colors[i + 1])
        #
        # ax1.set_xlabel('Effective Reproductive Number')
        # ax1.set_ylabel('Number of realizations')
        # ax1.legend()
        # fig.tight_layout()
        # core_name = f[12:-4]
        # os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
        # fig.savefig('hist_RU_' + str(q) + core_name + '_inf_eff.png', dpi=250)
        # os.chdir(wd)
        print('Plotting absolute histograms!')
        fig = plt.figure(figsize=(6, 3))
        ax1 = fig.add_subplot()

        minbin = 0.  # np.min([np.min(all_RUi_end_abs), np.min(all_RU_end_abs)])
        maxbin = 1.1 * np.max(all_RU_i_abs)
        step = (maxbin - minbin) / 20
        if step > 0.02:
            step = 0.02
        bins = np.arange(minbin, maxbin, step)

        for i in range(0, len(all_RU_i_abs)):
            # Initial R
            R_start_abs = np.array(all_RU_i_abs[i])

            # add histogram to plot
            ax1.hist(R_start_abs, bins=bins, label='T' + str(i + 1), \
                     density=False, color=colors[i + 1], zorder=3, edgecolor='black')
            # , lw=1, ec='w', alpha=1.
        ax1.set_xlabel('Effective Reproductive Number', fontsize=14)
        ax1.set_ylabel('Number of bootstraps', fontsize=14)
        ax1.legend(prop={'size': 12}, frameon=False)
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        plt.xlim([0, 3.5])
        ax1.patch.set_facecolor('lightgrey')
        plt.grid(color='white')
        fig.tight_layout()

        core_name = f[12:-4]
        os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
        fig.savefig('hist_R_eff.png', dpi=250)
        fig.savefig('hist_R_eff.pdf', format='pdf')
        os.chdir(wd)

    plt.close('all')

    return 0


def predict_VaccineOneGroup(parameters_fit, fixed_pars_in, frac_vaccinated_in, t, q, vaccineGroup, effVac):
    n_adm = fixed_pars_in[0]
    global constantMobility
    global constantR
    global fixedSocial

    fixedSocial = 0.5 * np.ones((n_adm,))
    fixedSocial[vaccineGroup - 1] = ((1 - frac_vaccinated_in) * 0.5 + frac_vaccinated_in * (1 - effVac)) / (
                1 - frac_vaccinated_in + frac_vaccinated_in * (1 - effVac))

    frac_vaccinated = frac_vaccinated_in * effVac

    # Change number of susceptibles and recovered
    n_sus = [[] for i in range(0, n_adm)]  # fixed_pars_in[2 * n_adm].copy()
    n_rec = fixed_pars_in[2 * n_adm][vaccineGroup - 1] * frac_vaccinated
    for i in range(0, n_adm):
        if i == vaccineGroup - 1:
            n_sus[i] = fixed_pars_in[2 * n_adm][i] * (1 - frac_vaccinated)
        else:
            n_sus[i] = fixed_pars_in[2 * n_adm][i]

            # intialize a single exposed case in tile 1

    pars_vac90 = parameters_fit.copy()
    pars_vac90[n_adm * 0:n_adm * 5] = 0
    pars_vac90[0] = 1
    pars_vac90[n_adm * 4 + vaccineGroup - 1] = n_rec

    # parameters
    fixed_pars_vac90 = fixed_pars_in[0:2 * n_adm].copy()
    fixed_pars_vac90.append(n_sus)
    fixed_pars_vac90 = fixed_pars_vac90 + fixed_pars_in[2 * n_adm + 1:len(fixed_pars_in)]

    # Predice
    constantMobility = True
    constantR = True
    fit_vaccine = solution_SEUI(t, pars_vac90, fixed_pars_vac90).T
    constantMobility = False
    constantR = False
    fixedSocial = 1

    n_U = fit_vaccine[2 * n_adm + q - 1]
    n_I = fit_vaccine[3 * n_adm + q - 1]
    n_Ui = fit_vaccine[4 * n_adm + q - 1]
    n_RUi = fit_vaccine[5 * n_adm + q - 1] - fit_vaccine[5 * n_adm + q - 1][0]

    return n_I + n_Ui + n_RUi


def predict_Vaccine(parameters_fit, fixed_pars, frac_vaccinated_in, t, q, effVac):
    n_adm = fixed_pars[0]
    global constantMobility
    global constantR
    global fixedSocial

    fixedSocial = ((1 - frac_vaccinated_in) * 0.5 + frac_vaccinated_in * (1 - effVac)) / (
                1 - frac_vaccinated_in + frac_vaccinated_in * (1 - effVac))

    frac_vaccinated = frac_vaccinated_in * effVac

    # intialize a single exposed case in tile 1
    pars_vac90 = parameters_fit.copy()
    pars_vac90[n_adm * 0:n_adm * 5] = 0
    pars_vac90[0] = 1
    pars_vac90[n_adm * 4:n_adm * 5] = np.array(fixed_pars[2 * n_adm]) * frac_vaccinated

    # parameters
    fixed_pars_vac90 = fixed_pars[0:2 * n_adm].copy()
    fixed_pars_vac90.append(list(np.array(fixed_pars[2 * n_adm]) * (1 - frac_vaccinated)))
    fixed_pars_vac90 = fixed_pars_vac90 + fixed_pars[2 * n_adm + 1:len(fixed_pars)]

    # Predict
    constantMobility = True
    constantR = True
    fit_vaccine = solution_SEUI(t, pars_vac90, fixed_pars_vac90).T
    constantMobility = False
    constantR = False
    fixedSocial = 1

    n_U = fit_vaccine[2 * n_adm + q - 1]
    n_I = fit_vaccine[3 * n_adm + q - 1]
    n_Ui = fit_vaccine[4 * n_adm + q - 1]
    n_RUi = fit_vaccine[5 * n_adm + q - 1] - fit_vaccine[5 * n_adm + q - 1][0]

    return n_I + n_Ui + n_RUi


def plot_SEUI_load(t_trn, real_cuminf_trn, mdl_cuminf, mdl_asi, mdl_e, pop, result, r2_inf, adm, core_name, t_RU, RU):
    fig1 = plt.figure(figsize=(12, 6))
    for i in range(2):
        ax = fig1.add_subplot(1, 2, i + 1)
        ax.plot(t_trn, real_cuminf_trn, 'gray', marker='o', markersize=3, \
                label='confirmed cases (data)')
        # ax.plot(t_trn,real_fat_trn,'red',marker='o',markersize=3,\
        #      label='deceased (data)')
        ax.plot(t_trn, mdl_cuminf[:len(t_trn)], 'gray', \
                label='confirmed infected cases (model)')
        ax.plot(t_trn, mdl_asi[:len(t_trn)], 'blue', \
                label='asymptomatic infecteous (model)')
        ax.plot(t_trn, mdl_e[:len(t_trn)], 'green', \
                label='exposed (model)')
        # ax.set_xticks(ticks)
        # ax.set_xticklabels(labels,rotation=45)
        plt.xlabel('Time [days]')
        plt.grid()
        if i == 1:
            plt.ylim([0, 1.05 * max(max(real_cuminf_trn), max(real_cuminf_trn))])
        elif i == 0:
            plt.legend()
            plt.ylim([1, pop])
            plt.yscale('log')
            plt.ylabel('Number of Cases')
    fig1.suptitle('Population Compartments in ' + str(adm) + ' - R2 = ' + \
                  '{:.3f}'.format(r2_inf))
    fig1.savefig(str(adm) + core_name + '_inf.png', dpi=250)

    plt.show()

    fig2 = plt.figure(figsize=(12, 6))

    # ax1  = fig2.add_subplot(121)
    # ax1.plot(t_trn,real_fat_trn,'red',marker='o',label='deceased (data)')
    # ax1.plot(t_trn,mdl_fat[:len(t_trn)],'r',label='deceased (model)')
    # ax1.plot(t_tst,mdl_fat[len(t_trn):],'black' )
    # ax1.plot(t_tst,real_fat_tst,'ko' )
    # # ax1.set_xticks(ticks)
    # # ax1.set_xticklabels(labels,rotation=45)
    # plt.grid()
    # plt.legend()
    # plt.ylabel('Number of Dead')
    # plt.xlabel('Time [days]')
    ax2 = fig2.add_subplot(121)
    ax2.plot(t_RU, RU, label='Reproductive number before symptoms')
    # ax2.set_xticks(ticks)
    # ax2.set_xticklabels(labels,rotation=45)
    plt.xlabel('Time [days]')
    plt.legend()
    fig2.suptitle('R in ' + str(adm))
    fig2.savefig(str(adm) + core_name + '_R.png', dpi=250)

    return 0


def eval_randomTiles():  #

    run_ID_prefix = 'Final_uncert_20201204_random'
    n_adm = 3

    R1_abs = []
    R2_abs = []
    R3_abs = []
    R1_eff = []
    R2_eff = []
    R3_eff = []
    for r in range(1, 34):
        if r < 10:
            run_ID_suffix = '00' + str(r)
        else:
            run_ID_suffix = '0' + str(r)
        run_ID = run_ID_prefix + run_ID_suffix

        # Load results
        folder = os.path.join(wd, 'Results', run_ID, 'parameters')
        files = os.listdir(folder)
        for i, f in enumerate(files):
            if f[-9:] == 'itted.pkl':
                infile = open(os.path.join(folder, f), 'rb')
                result = pickle.load(infile)
                infile.close()

            elif f[-9:] == 'fixed.pkl':
                infile = open(os.path.join(folder, f), 'rb')
                fixed_pars = pickle.load(infile)
                infile.close()
                Adj = fixed_pars[3]

        # Save to d
        for q in range(0, n_adm):

            norm = Adj.values[q, :].sum()

            if useMultiplicModel:
                if q > 0:
                    R = result[0].x[5 * n_adm + q] * result[0].x[5 * n_adm + 0]
                else:
                    R = result[0].x[5 * n_adm + q]
            else:
                R = result[0].x[5 * n_adm + q]

            if q == 0:
                R1_abs.append(R)
                R1_eff.append(R * norm)
            elif q == 1:
                R2_abs.append(R)
                R2_eff.append(R * norm)
            else:
                R3_abs.append(R)
                R3_eff.append(R * norm)

    df = pd.DataFrame(list(zip(R1_abs, R2_abs, R3_abs, R1_eff, R2_eff, R3_eff)),
                      columns=['R1_abs', 'R2_abs', 'R3_abs', 'R1_eff', 'R2_eff', 'R3_eff'])
    df.to_csv(os.path.join(wd, 'Results', run_ID_prefix + '_random.csv'))

    return 0