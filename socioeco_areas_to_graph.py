#!/usr/bin/env python3
#
# This is a simple copmpartmental epidemiology model (SEIRD-model) describing
# the evolution of susceptible, exposed, infected, recovered and deceased
# population sizes.

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


localMac = False
wd = os.getcwd()
r2gof_threshRsq = 0.
useForTiles = 'MedianIncome2017'  # Choose from: 'LivingSpace', 'SENIOR_ANT','1PHouseholds','MedianIncome2017', 'random' ,'CoHab_index',#
useForTilesList = ['LivingSpace', '1PHouseholds', 'SENIOR_ANT',
                   'MedianIncome2017']  # ,'Vollzeitaequivalent' 'SENIOR_ANT'] #, 'MedianIncome2017'
useMultipeSeparations = False
randomIndex = '001'
n_jobs = 30
n_uncert = 60  # Number of runs to test uncertainty
diagonalMobilityZero = False
use7DayAV = True
RperTilefix = True
useRelativeR = True
useMultiplicModel = True
useReortingDelay = False
useSame_ab = True
useSigmoid = False
useSymMat = True
useStretchSocial = False
useHomeReproductive = False

useImputation = True

useExponentTimeseries = False
useVariable_p_unr = False
useNoUi = False
multiplyCaseNUmbers = False

global constantMobility
constantMobility = False

global zeroMobility
zeroMobility = False

global constantR
constantR = False

global fixedSocial
fixedSocial = 1

useNaturalBreaks = False
useGradient = False

global time_dep_soc


### MAIN #####_##################################################################

def main():
    global randomIndex

    for r in range(0, 1):
        randomIndex = '00' + str(r)

        # general setup
        adm0 = 'CHE'  # iso3 pcode of the country of interest
        lvl = 1  # level of analysis (0:national,1:state,...)
        mode = 'a'  # s: single, a: all that are listed simulataneously (disease sp. param. same for all)
        coupled = True  # switches coupling on and off
        local = True  # switches locality of parameters on and off
        useSoc = False  # use the social graph structure in addition to geographical
        start = dt.date(2020, 2, 22)  # 26   # start date of analysis (CHE:2/23,DEU:1/28) - 3,7
        end = dt.date(2020, 4, 22)  # end date of analysis
        t_meas = 1000  # postulated time of government measures
        n_tst = 63  # 57                      # number of days reserved for testing
        model = 'SEUI'  # the ODE model used: SEUI or SEAIRD
        all_quat = list(np.arange(1,
                                  21))  # the summarized quaters of Basel to be analysed: choose from 1-20,'all', or list(np.arange(1,21),30)
        neigh = 'transport'  # Define which neighborhood to use: 'physical','estimate','transport'
        alpha_fix = True  # If true: use the same mobility rate for all quarters. If false: R const, alpha varies
        uncert = True  # Do a crude uncertainty estimation
        run_ID = 'Test_' + useForTiles  # Name for this specific run - folder will be created in Results
        useparallel = True  # number of parallel processes (for uncertainty analysis)
        saveDataUnc = False  # Save the data generated in this uncertainty run for future reference
        loadDataUnc = False  # Whether the data for the uncertainty runs should be loaded or generated
        uncData_id = 'start2020-3-7_end2020-4-22'  # ID from which to load - or to which to save the uncertainty data
        delta_t = 14  # time difference to the 7.3.2020 which is the date for which the time series data is normalized

        print('RUNNING: ' + run_ID)

        # transport graphs data from SBB etc.
        loadFr_dyn = True  # load a specific time series for alpha
        useSigTran = True
        useSepTimeForRandAlpha = True

        # Include unreported cases
        useUnrepor = True

        # Lineage information
        usePangolin = True
        pangolingStrain = 'yes'

        # Seed only a single case (exposed) - all other quarters start from 0!
        seedSingle = True
        seedQuarter = 1  # 9    # The quarter with the first case
        if useForTiles == '1PHouseholds' or useForTiles == 'Vollzeitaequivalent':
            seedQuarter = 2

        # Socioeconomic 'quarters'
        mergeQuatSoc = True
        n_splitsSoc = 3  # Choose from 3 to 9, one section for 'NaN'
        if mergeQuatSoc:
            all_quat = list(np.arange(1, n_splitsSoc + 1))
            neigh = 'merged'

        # initial values and constaints - separate depending on model chosen
        # 4 COMPARTMENTS
        R_infU_in_0 = 7.88  # 2.  # transmission rate caused by symptomatic infecteous cases
        T_infU_0 = 3.  # length of infectious period in days fot those who recover
        T_inc_0 = 3.  # duration of incubation period in days (global)
        b_deR_0 = 3.  # 2.#0.2039#2.  # speed of change (in days) of R_asy
        a_deR_0 = 23  # delta_t+9.   #11.6  # time of change (in days) of R_asy
        R_redU_0 = 2.  # 0.2966#.5  # reduced transmission/mobility rate
        bnd_R_infU_in = ((0.1, 40.),)  # ((1., 5.),)
        bnd_T_infU = ((2.1, 2.1),)
        bnd_T_inc = ((2., 2.),)
        bnd_R_redU = ((0., 5.),)
        bnd_b_deR = ((2., 10.),)  # ((2., 5),)#((0.2039, 0.2039),)##((1., 5.),)
        bnd_a_deR = ((0., 50.),)  # ((delta_t+9.-15, delta_t+9.+9),)#((9., 9.),)#((8., 20.),)

        # optional unreported compartment and recovered from it - a,b and Rred are assumed to be the same as for reported cases
        R_i_asy = .7  # Reproductive number for asymptomatic cases
        T_i_asy = 1.768
        p_asy = 0.88
        n_un_i_0 = .1  # initially infecteous cases who will die or recover
        n_un_r_0 = .1  # initially infecteous who will recover
        bnd_n_uni = ((0, 8),)
        bnd_n_unr = ((0, 2),)
        bnd_R_i_asy = ((0.6, 1.),)  # ((1., 5.),)
        bnd_T_i_asy = ((1.768, 1.768),)  # ((2., 12.),)
        bnd_p_asy = ((0.88, 0.88),)

        # Mobility and connectivity
        alpha_0 = 1.  # 0.05  # initial mobility rate
        alpha_f = .01  # final mobility rate
        a_alpha = 11.6  # time of change (in days) mobility rate
        b_alpha = .2  # speed of change (in days) of mobility rate
        adj_el = 0.5  # element of the adjaceny matrix
        bnd_alpha = ((1., 1.),)  # ((0.01, 3.7),)
        bnd_alpha_f = ((0., 1.),)
        bnd_a_alpha = ((8, 20),)
        bnd_b_alpha = ((0.1, 2),)
        bnd_adj_el = ((0., 1.),)

        # Social graph
        alphaS = 1.  # social mobility rate
        bnd_alphaS = ((0., 1.),)

        # 8 COMPARTMENTS
        T_infI_0 = 15  # length of infectious period in days for those who die
        p_fat_0 = .05  # fatality rate
        R_red_0 = 0.1  # reduced transmission of asympt. inf. cases after measure
        p_sym_0 = 1  # probability to be symptomatic
        R_asy_in_0 = 1  # initial transmission rate for asympt. infecteous cases
        T_infA_0 = 15  # length og infectious period in asymptomatic cases
        bnd_T_infI = ((1, 30),)
        bnd_p_fat = ((0, .15),)
        bnd_R_red = ((0, 0.2),)
        bnd_p_sym = ((1, 1),)
        bnd_R_asy_in = ((1, 3),)
        bnd_T_infA = ((1, 30),)

        # Initial conditions
        # 4 COMPARTMENTS
        n_exp_0 = 1  # initially exposed cases
        n_inf_0 = .1  # initially infecteous cases who will die or recover
        n_und_0 = .1  # initially infecteous who will recover
        bnd_n_exp = ((0, 8),)
        bnd_n_inf = ((0, 2),)
        bnd_n_und = ((0, 8),)

        # 8 COMPARTMENTS
        n_asi_0 = 0  # initially asymptomatic infecteous cases
        n_asr_0 = 0  # initially asymptomatic recovered cases
        n_rec_0 = 0  # initially recovered cases
        n_fat_0 = 0  # initially deceased cases
        bnd_n_rec = ((0, 0),)
        bnd_n_fat = ((0, 0),)
        bnd_n_asi = ((0, 0),)
        bnd_n_asr = ((0, 0),)

        # Make output director
        newdir = wd + '/Results/' + run_ID
        if useMultipeSeparations:
            if not os.path.exists(newdir):
                os.mkdir(newdir)
                os.chdir(newdir)
                for i in useForTilesList:
                    os.mkdir(i)
                    os.chdir(i)
                    os.mkdir('figures')
                    os.mkdir('fitting')
                    os.mkdir('original')
                    os.mkdir('parameters')
                    os.chdir(newdir)

        else:
            if not os.path.exists(newdir):
                os.mkdir(newdir)
                os.chdir(newdir)
                os.mkdir('figures')
                os.mkdir('fitting')
                os.mkdir('original')
                os.mkdir('parameters')
                os.chdir(wd)

        if mode == 's':
            for i, q in enumerate(all_quat):
                quat = [q]

                run_model(quat, adm0, newdir, lvl, coupled, local, t_meas, n_tst, start, end, mode, model,
                          neigh, alpha_fix, uncert, n_uncert, run_ID, useparallel, n_exp_0, n_inf_0, n_rec_0,
                          n_fat_0, n_asi_0, n_asr_0, n_und_0, R_infU_in_0, T_infI_0, T_inc_0, p_fat_0,
                          R_red_0, R_redU_0, p_sym_0, R_asy_in_0, b_deR_0, a_deR_0, T_infA_0, T_infU_0,
                          alpha_0, alpha_f, a_alpha, b_alpha, adj_el, bnd_n_exp, bnd_n_inf, bnd_n_rec,
                          bnd_n_fat, bnd_n_asi, bnd_n_asr, bnd_n_und, bnd_R_infU_in, bnd_T_infI, bnd_T_inc,
                          bnd_p_fat, bnd_R_red, bnd_R_redU, bnd_p_sym, bnd_R_asy_in, bnd_b_deR, bnd_a_deR,
                          bnd_T_infA, bnd_T_infU, bnd_alpha, bnd_alpha_f, bnd_a_alpha, bnd_b_alpha, bnd_adj_el,
                          saveDataUnc, loadDataUnc, uncData_id, alphaS, bnd_alphaS, useSoc, loadFr_dyn, useSigTran,
                          R_i_asy, T_i_asy, p_asy, n_un_i_0, n_un_r_0, bnd_n_uni, bnd_n_unr, bnd_R_i_asy, bnd_T_i_asy,
                          bnd_p_asy,
                          useUnrepor, usePangolin, pangolingStrain, mergeQuatSoc, n_splitsSoc, seedSingle, seedQuarter,
                          useSepTimeForRandAlpha, delta_t)

        else:
            quat = all_quat

            if useMultipeSeparations:
                run_model_parallel(quat, adm0, newdir, lvl, coupled, local, t_meas, n_tst, start, end, mode, model,
                                   neigh, alpha_fix, uncert, n_uncert, run_ID, useparallel, n_exp_0, n_inf_0, n_rec_0,
                                   n_fat_0, n_asi_0, n_asr_0, n_und_0, R_infU_in_0, T_infI_0, T_inc_0, p_fat_0,
                                   R_red_0, R_redU_0, p_sym_0, R_asy_in_0, b_deR_0, a_deR_0, T_infA_0, T_infU_0,
                                   alpha_0, alpha_f, a_alpha, b_alpha, adj_el, bnd_n_exp, bnd_n_inf, bnd_n_rec,
                                   bnd_n_fat, bnd_n_asi, bnd_n_asr, bnd_n_und, bnd_R_infU_in, bnd_T_infI, bnd_T_inc,
                                   bnd_p_fat, bnd_R_red, bnd_R_redU, bnd_p_sym, bnd_R_asy_in, bnd_b_deR, bnd_a_deR,
                                   bnd_T_infA, bnd_T_infU, bnd_alpha, bnd_alpha_f, bnd_a_alpha, bnd_b_alpha, bnd_adj_el,
                                   saveDataUnc, loadDataUnc, uncData_id, alphaS, bnd_alphaS, useSoc, loadFr_dyn,
                                   useSigTran,
                                   R_i_asy, T_i_asy, p_asy, n_un_i_0, n_un_r_0, bnd_n_uni, bnd_n_unr, bnd_R_i_asy,
                                   bnd_T_i_asy,
                                   bnd_p_asy,
                                   useUnrepor, usePangolin, pangolingStrain, mergeQuatSoc, n_splitsSoc, seedSingle,
                                   seedQuarter,
                                   useSepTimeForRandAlpha, delta_t)
            else:
                run_model(quat, adm0, newdir, lvl, coupled, local, t_meas, n_tst, start, end, mode, model,
                          neigh, alpha_fix, uncert, n_uncert, run_ID, useparallel, n_exp_0, n_inf_0, n_rec_0,
                          n_fat_0, n_asi_0, n_asr_0, n_und_0, R_infU_in_0, T_infI_0, T_inc_0, p_fat_0,
                          R_red_0, R_redU_0, p_sym_0, R_asy_in_0, b_deR_0, a_deR_0, T_infA_0, T_infU_0,
                          alpha_0, alpha_f, a_alpha, b_alpha, adj_el, bnd_n_exp, bnd_n_inf, bnd_n_rec,
                          bnd_n_fat, bnd_n_asi, bnd_n_asr, bnd_n_und, bnd_R_infU_in, bnd_T_infI, bnd_T_inc,
                          bnd_p_fat, bnd_R_red, bnd_R_redU, bnd_p_sym, bnd_R_asy_in, bnd_b_deR, bnd_a_deR,
                          bnd_T_infA, bnd_T_infU, bnd_alpha, bnd_alpha_f, bnd_a_alpha, bnd_b_alpha, bnd_adj_el,
                          saveDataUnc, loadDataUnc, uncData_id, alphaS, bnd_alphaS, useSoc, loadFr_dyn, useSigTran,
                          R_i_asy, T_i_asy, p_asy, n_un_i_0, n_un_r_0, bnd_n_uni, bnd_n_unr, bnd_R_i_asy, bnd_T_i_asy,
                          bnd_p_asy,
                          useUnrepor, usePangolin, pangolingStrain, mergeQuatSoc, n_splitsSoc, seedSingle, seedQuarter,
                          useSepTimeForRandAlpha, delta_t)

                # Plot
                # try:
                main_eval(run_ID)
                # except:
                #     print('Failed at plotting!')

    return None


### FUNCTIONS #################################################################
def run_model_parallel(quat, adm0, newdir, lvl, coupled, local, t_meas, n_tst, start, end, mode, model,
                       neigh, alpha_fix, uncert, n_uncert, run_ID, useparallel,
                       n_exp_0, n_inf_0, n_rec_0, n_fat_0, n_asi_0, n_asr_0, n_und_0, R_infU_in_0,
                       T_infI_0, T_inc_0, p_fat_0, R_red_0, R_redU_0, p_sym_0, R_asy_in_0, b_deR_0,
                       a_deR_0, T_infA_0, T_infU_0, alpha_0, alpha_f, a_alpha, b_alpha, adj_el,
                       bnd_n_exp, bnd_n_inf, bnd_n_rec, bnd_n_fat, bnd_n_asi, bnd_n_asr,
                       bnd_n_und, bnd_R_infU_in, bnd_T_infI, bnd_T_inc, bnd_p_fat, bnd_R_red, bnd_R_redU,
                       bnd_p_sym, bnd_R_asy_in, bnd_b_deR, bnd_a_deR, bnd_T_infA, bnd_T_infU, bnd_alpha,
                       bnd_alpha_f, bnd_a_alpha, bnd_b_alpha, bnd_adj_el, saveDataUnc, loadDataUnc, uncData_id,
                       alphaS_0, bnd_alphaS, useSoc, loadFr_dyn, useSigTran,
                       R_i_asy, T_i_asy, p_asy, n_un_i_0, n_un_r_0, bnd_n_uni, bnd_n_unr, bnd_R_i_asy, bnd_T_i_asy,
                       bnd_p_asy,
                       useUnrepor, usePangolin, pangolingStrain, mergeQuatSoc, n_splitsSoc, seedSingle, seedQuarter,
                       useSepTimeForRandAlpha, delta_t):
    # Correct bounds for some options
    if not useUnrepor:
        R_i_asy = 0  # Reproductive number for asymptomatic cases
        T_i_asy = 0
        p_asy = 0
        n_un_i_0 = 0  # initially infecteous cases who will die or recover
        n_un_r_0 = 0  # initially infecteous who will recover
        bnd_n_uni = ((0, 0),)
        bnd_n_unr = ((0, 0),)
        bnd_R_i_asy = ((0, 0),)  # ((1., 5.),)
        bnd_T_i_asy = ((0, 0),)

    global useForTilesList
    global useForTiles

    all_params = []
    all_fixed_pars = []
    all_bnds = []
    all_t_trn = []
    all_t_tst = []
    all_data_trn = []
    all_data_tst = []

    for i_tiles, tile in enumerate(useForTilesList):
        useForTiles = tile

        # corresponding setup details
        n_cmp, A, n_adm, t_trn, t_tst, data_trn, data_tst, pop, ASoc, time_dep, data_trn_abs = \
            setup_ILGE(lvl, coupled, local, t_meas, n_tst, start, end, mode, quat, model, neigh, useSoc, loadFr_dyn,
                       usePangolin, pangolingStrain, mergeQuatSoc, n_splitsSoc, useUnrepor, delta_t)

        # assemble parameter list to be passed to optimizer
        if tile == '1PHouseholds' or tile == 'Vollzeitaequivalent':
            seedQuarter = 2
        else:
            seedQuarter = 1
        ind_seedQuarter = quat.index(seedQuarter)

        par_list = par_list_general(model, n_adm, n_exp_0, n_inf_0, n_rec_0, n_fat_0, n_asi_0,
                                    n_asr_0, n_und_0, R_infU_in_0, T_infI_0, T_inc_0, p_fat_0, R_red_0, R_redU_0,
                                    p_sym_0, R_asy_in_0, b_deR_0, a_deR_0, T_infA_0, T_infU_0, alpha_0, alpha_f,
                                    a_alpha, b_alpha, adj_el, local, neigh, alpha_fix, alphaS_0,
                                    R_i_asy, T_i_asy, p_asy, n_un_i_0, n_un_r_0, seedSingle, ind_seedQuarter)

        # assemble optimization constraints
        bnds = bnds_vars_general(n_adm, model, bnd_n_exp, bnd_n_inf, bnd_n_rec, bnd_n_fat, bnd_n_asi, bnd_n_asr,
                                 bnd_n_und, bnd_R_infU_in, bnd_T_infI, bnd_T_inc, bnd_p_fat, bnd_R_red, bnd_R_redU,
                                 bnd_p_sym, bnd_R_asy_in, bnd_b_deR, bnd_a_deR, bnd_T_infA, bnd_T_infU, bnd_alpha,
                                 bnd_alpha_f, bnd_a_alpha, bnd_b_alpha, bnd_adj_el, local, neigh, alpha_fix, bnd_alphaS,
                                 bnd_n_uni, bnd_n_unr, bnd_R_i_asy, bnd_T_i_asy, bnd_p_asy, seedSingle, ind_seedQuarter)

        # fixed paramters
        if len(t_tst) > 0:
            t_max = np.max([np.max(t_trn), np.max(t_tst)])
        else:
            t_max = np.max(t_trn)
        fixed_pars = [n_adm, n_cmp, t_meas, A, int(coupled), local, pop, neigh, alpha_fix, ASoc, loadFr_dyn, time_dep,
                      useSigTran, useUnrepor, useSepTimeForRandAlpha, delta_t, t_max]
        print(useForTiles + ' population: ' + str(pop))

        # Summarize:
        all_params.append(par_list)
        all_fixed_pars.append(fixed_pars)
        all_bnds.append(bnds)
        all_t_trn.append(t_trn)
        all_data_trn.append(data_trn)
        all_data_tst.append(data_tst)
        all_t_tst.append(t_tst)

    # fit model
    all_result_i, all_t_i, all_fit_i, all_params_out, all_fixed_pars_out = fit_general_parallel(all_params,
                                                                                                all_data_trn, all_bnds,
                                                                                                all_fixed_pars, adm0,
                                                                                                all_t_trn, all_t_tst)
    print(all_result_i.x)

    # Save for each subset
    for i_tiles, tile in enumerate(useForTilesList):
        # Plot
        main_eval_parallel(run_ID, tile, all_t_i[i_tiles], all_fit_i[i_tiles], all_params_out[i_tiles],
                           all_fixed_pars_out[i_tiles], all_data_trn[i_tiles], all_t_trn[i_tiles])

    return 0


# Fit parallel
def fit_general_parallel(all_par_list, all_data_trn, all_bnds, all_fixed_pars, adm0, all_t_trn, all_t_tst):
    result, n_parameters = dofit_SEUI_parallel(all_par_list, all_data_trn, all_bnds, all_fixed_pars, adm0)

    # optain curve resulting from optimization
    n_separations = len(all_par_list)
    n_adm = all_fixed_pars[0][0]
    params = result.x[5 * n_separations * n_adm:]
    inits = result.x[:5 * n_separations * n_adm]
    all_fit = []
    all_t = []
    all_params = []
    for i in range(0, n_separations):

        t = np.concatenate((all_t_trn[i], all_t_tst[i]))
        this_par = params[:n_parameters]

        if i > 0:
            # b
            this_par[2 * n_adm + 1:3 * n_adm + 1] = params[(i - 1) * 4 * n_adm + n_parameters:(
                                                                                                          i - 1) * 4 * n_adm + n_parameters + n_adm]

            # a
            this_par[3 * n_adm + 1:4 * n_adm + 1] = params[(i - 1) * 4 * n_adm + n_parameters + n_adm:(
                                                                                                                  i - 1) * 4 * n_adm + n_parameters + 2 * n_adm]

            # Rinit
            this_par[:n_adm] = params[(i - 1) * 4 * n_adm + n_parameters + 2 * n_adm:(
                                                                                                 i - 1) * 4 * n_adm + n_parameters + 3 * n_adm]

            # frac Rend
            this_par[n_adm + 1:(n_adm + 1) + n_adm] = params[(i - 1) * 4 * n_adm + n_parameters + 3 * n_adm:(
                                                                                                                        i - 1) * 4 * n_adm + n_parameters + 4 * n_adm]

        this_init = inits[i * n_adm * 5:(i + 1) * n_adm * 5]
        pars_use = np.array(list(this_init) + list(this_par))
        fit = solution_SEUI(t, pars_use, all_fixed_pars[i]).T
        all_fit.append(fit)
        all_t.append(t)
        all_params.append(pars_use)

    return result, all_t, all_fit, all_params, all_fixed_pars


def dofit_SEUI_parallel(all_par_list, all_data_train, all_bnds, all_fixed_pars, adm0):
    fixed_pars_list = all_fixed_pars[0]
    fixed_pars_list.append(len(all_par_list))  # save number of separations
    n_adm = fixed_pars_list[0]
    n_cmp = fixed_pars_list[1]

    par_list = all_par_list[0][n_adm * (n_cmp - 1):]
    n_parameters = len(par_list)
    bnds_list = list(all_bnds[0])[n_adm * (n_cmp - 1):]
    startCon_list = all_par_list[0][:n_adm * (n_cmp - 1)]
    startCon_bnds_list = list(all_bnds[0])[:n_adm * (n_cmp - 1)]
    for i in range(1, len(all_par_list)):
        # get parameter set for this separation
        pars = all_par_list[i]
        bnds = all_bnds[i]
        fixed_pars = all_fixed_pars[i]

        # Other fit parameters specific to each separation
        b_deR = pars[n_adm * (n_cmp - 1) + 2 * n_adm + 1:n_adm * (n_cmp - 1) + 3 * n_adm + 1]
        a_deR = pars[n_adm * (n_cmp - 1) + 3 * n_adm + 1:n_adm * (n_cmp - 1) + 4 * n_adm + 1]
        R_infU_in = pars[n_adm * (n_cmp - 1):n_adm * (n_cmp - 1) + n_adm]
        R_redU_frac = pars[n_adm * (n_cmp - 1) + n_adm + 1:n_adm * (n_cmp - 1) + (n_adm + 1) + n_adm]
        startingCond = pars[:n_adm * (n_cmp - 1)]
        startCon_list = startCon_list + startingCond
        par_list = par_list + b_deR + a_deR + R_infU_in + R_redU_frac

        bnds_b_deR = list(bnds[n_adm * (n_cmp - 1) + 2 * n_adm + 1:n_adm * (n_cmp - 1) + 3 * n_adm + 1])
        bnds_a_deR = list(bnds[n_adm * (n_cmp - 1) + 3 * n_adm + 1:n_adm * (n_cmp - 1) + 4 * n_adm + 1])
        bnds_R_infU_in = list(bnds[n_adm * (n_cmp - 1):n_adm * (n_cmp - 1) + n_adm])
        bnds_R_redU_frac = list(bnds[n_adm * (n_cmp - 1) + n_adm + 1:n_adm * (n_cmp - 1) + (n_adm + 1) + n_adm])
        bnds_list = bnds_list + bnds_b_deR + bnds_a_deR + bnds_R_infU_in + bnds_R_redU_frac
        startCon_bnds_list = startCon_bnds_list + list(bnds[:n_adm * (n_cmp - 1)])

        # Optionally fit adjacency matrix
        Adj = fixed_pars[3]
        fixed_pars_list.append(Adj)
        fixed_pars_list.append(fixed_pars[6])

    # fixed_pars_list.append(len(par_list))
    par_list = startCon_list + par_list
    bnds_list = startCon_bnds_list + bnds_list
    bnds_tpl = tuple(bnds_list)
    fixed_pars_list.append(n_parameters)

    result = \
        minimize(lambda var1, var2: residual_SEUI_parallel(var1, var2, fixed_pars_list, adm0), \
                 par_list, \
                 args=(all_data_train), \
                 method='L-BFGS-B', \
                 bounds=bnds_tpl, \
                 # tol=1e-5, \
                 options={'gtol': 1e-8, 'ftol': 1e-8, 'disp': True, 'maxiter': 200})

    with open(str(useForTilesList) + '_results.pkl', 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    return result, n_parameters


def solution_SEUI_parallel(t, par_list, fixed_pars):
    '''Numerical ODE solution for given set of parameters.

    Parameters
    ----------
    t          : time points vector
    par_list   : initial conditions and AFTER the initial parameters values.
    fixed_pars : fixed parameters

    Returns
    -------
    numerical solution of the ODE system function
    '''
    n_adm = fixed_pars[0]
    n_cmp = fixed_pars[1]
    n_separations = fixed_pars[17]

    # Initial conditions
    for i in range(0, n_separations):

        if i == 0:
            s_0 = fixed_pars[6]
            y0 = np.concatenate((s_0, par_list[:((n_cmp - 1) * n_adm)]))
        else:
            s_0 = fixed_pars[19 + (i - 1) * 2]
            y0 = np.concatenate((y0, s_0, par_list[i * ((n_cmp - 1) * n_adm):(i + 1) * ((n_cmp - 1) * n_adm)]))

    pars = par_list[n_separations * (n_cmp - 1) * n_adm:]

    sol = odeint(lambda a, b, c: ode_model_SEUI_parallel(a, b, c, fixed_pars), y0, t, args=(pars,))

    return sol


def run_model(quat, adm0, newdir, lvl, coupled, local, t_meas, n_tst, start, end, mode, model,
              neigh, alpha_fix, uncert, n_uncert, run_ID, useparallel,
              n_exp_0, n_inf_0, n_rec_0, n_fat_0, n_asi_0, n_asr_0, n_und_0, R_infU_in_0,
              T_infI_0, T_inc_0, p_fat_0, R_red_0, R_redU_0, p_sym_0, R_asy_in_0, b_deR_0,
              a_deR_0, T_infA_0, T_infU_0, alpha_0, alpha_f, a_alpha, b_alpha, adj_el,
              bnd_n_exp, bnd_n_inf, bnd_n_rec, bnd_n_fat, bnd_n_asi, bnd_n_asr,
              bnd_n_und, bnd_R_infU_in, bnd_T_infI, bnd_T_inc, bnd_p_fat, bnd_R_red, bnd_R_redU,
              bnd_p_sym, bnd_R_asy_in, bnd_b_deR, bnd_a_deR, bnd_T_infA, bnd_T_infU, bnd_alpha,
              bnd_alpha_f, bnd_a_alpha, bnd_b_alpha, bnd_adj_el, saveDataUnc, loadDataUnc, uncData_id,
              alphaS_0, bnd_alphaS, useSoc, loadFr_dyn, useSigTran,
              R_i_asy, T_i_asy, p_asy, n_un_i_0, n_un_r_0, bnd_n_uni, bnd_n_unr, bnd_R_i_asy, bnd_T_i_asy, bnd_p_asy,
              useUnrepor, usePangolin, pangolingStrain, mergeQuatSoc, n_splitsSoc, seedSingle, seedQuarter,
              useSepTimeForRandAlpha, delta_t):
    # Correct bounds for some options
    if not useUnrepor:
        R_i_asy = 0  # Reproductive number for asymptomatic cases
        T_i_asy = 0
        p_asy = 0
        n_un_i_0 = 0  # initially infecteous cases who will die or recover
        n_un_r_0 = 0  # initially infecteous who will recover
        bnd_n_uni = ((0, 0),)
        bnd_n_unr = ((0, 0),)
        bnd_R_i_asy = ((0, 0),)  # ((1., 5.),)
        bnd_T_i_asy = ((0, 0),)

    # corresponding setup details
    n_cmp, A, n_adm, t_trn, t_tst, data_trn, data_tst, pop, ASoc, time_dep, data_trn_abs = \
        setup_ILGE(lvl, coupled, local, t_meas, n_tst, start, end, mode, quat, model, neigh, useSoc, loadFr_dyn,
                   usePangolin, pangolingStrain, mergeQuatSoc, n_splitsSoc, useUnrepor, delta_t)

    # assemble parameter list to be passed to optimizer
    ind_seedQuarter = quat.index(seedQuarter)
    par_list = par_list_general(model, n_adm, n_exp_0, n_inf_0, n_rec_0, n_fat_0, n_asi_0,
                                n_asr_0, n_und_0, R_infU_in_0, T_infI_0, T_inc_0, p_fat_0, R_red_0, R_redU_0,
                                p_sym_0, R_asy_in_0, b_deR_0, a_deR_0, T_infA_0, T_infU_0, alpha_0, alpha_f,
                                a_alpha, b_alpha, adj_el, local, neigh, alpha_fix, alphaS_0,
                                R_i_asy, T_i_asy, p_asy, n_un_i_0, n_un_r_0, seedSingle, ind_seedQuarter)

    # assemble optimization constraints
    bnds = bnds_vars_general(n_adm, model, bnd_n_exp, bnd_n_inf, bnd_n_rec, bnd_n_fat, bnd_n_asi, bnd_n_asr,
                             bnd_n_und, bnd_R_infU_in, bnd_T_infI, bnd_T_inc, bnd_p_fat, bnd_R_red, bnd_R_redU,
                             bnd_p_sym, bnd_R_asy_in, bnd_b_deR, bnd_a_deR, bnd_T_infA, bnd_T_infU, bnd_alpha,
                             bnd_alpha_f, bnd_a_alpha, bnd_b_alpha, bnd_adj_el, local, neigh, alpha_fix, bnd_alphaS,
                             bnd_n_uni, bnd_n_unr, bnd_R_i_asy, bnd_T_i_asy, bnd_p_asy, seedSingle, ind_seedQuarter)

    # fixed paramters
    if len(t_tst) > 0:
        t_max = np.max([np.max(t_trn), np.max(t_tst)])
    else:
        t_max = np.max(t_trn)
    fixed_pars = [n_adm, n_cmp, t_meas, A, int(coupled), local, pop, neigh, alpha_fix, ASoc, loadFr_dyn, time_dep,
                  useSigTran, useUnrepor, useSepTimeForRandAlpha, delta_t, t_max]

    print(bnds)
    print(par_list)

    # Fit (optionally with uncertainty in data)
    result = []
    fit = []
    data_in = [data_trn]
    t_in = [t_trn]
    if uncert:

        # Fit (optionally with uncertainty in data)
        t_in = [t_trn]
        data_in = [data_trn]
        r2s = [0]

        if loadDataUnc:

            # Run with undisturbed data
            data_in = loadUncertData(n_uncert, quat, uncData_id, t_trn)

            # Uncertainty estimation
            input = [data_in, t_trn, model, par_list, bnds, fixed_pars, adm0, t_tst, neigh, alpha_fix]
            if not useparallel:
                for i in range(0, n_uncert + 1):
                    # Randomly shift data
                    out = uncertFit_loaded(input + [i])

                    data_in[i] = out[1]
                    result[i] = out[0]
                    fit[i] = out[2]

            else:
                input_par = [input + [0]]
                for i2 in range(1, n_uncert + 1):
                    input_par.append(input + [i2])

                outs = Parallel(n_jobs=n_uncert + 1, verbose=1, backend="multiprocessing")(
                    map(delayed(uncertFit_loaded), input_par))

                for i2 in range(0, n_uncert + 1):
                    out = outs[i2 - 1]
                    data_in[i2] = out[1]
                    result[i2] = out[0]
                    fit[i2] = out[2]

            if saveDataUnc:
                df_data = pd.DataFrame(data=data_in)
                df_data.to_csv(os.path.join(wd, 'Data', str(quat) + '_data.csv'))
        else:

            # First run once with undisturbed data
            result_i, t_i, fit_i = fit_general(model, par_list, data_in[0], bnds, fixed_pars, adm0, t_in[0],
                                               t_tst, neigh, alpha_fix)
            result = [result_i]
            fit = [fit_i]

            inds_g30 = [[] for j in range(0, n_adm)]
            inds_s10 = [[] for j in range(0, n_adm)]
            for i in range(0, n_adm):
                inds_g30[i] = np.where(data_in[0][1 + i] > 30)[0]
                inds_s10[i] = np.where(data_in[0][1 + i] <= 30)[0]
                if len(inds_g30[i]) == 0:
                    inds_g30[i] = np.arange(len(data_in[0][1 + i]) - 5, len(data_in[0][1 + i]))
                    inds_s10[i] = np.arange(0, len(data_in[0][1 + i]) - 5)

            # tmp = data_in[0][1][inds_g30]
            # tmp[tmp == 0] = 1.
            # rmse = np.mean((data_in[0][1][inds_g30] -
            #                 fit_i[3 * n_adm:4 * n_adm][0][inds_g30[0]:len(data_in[0][1])]) ** 2
            #                / tmp)
            # if rmse<0.1 or rmse>0.5:
            #     rmse = 0.1
            rmse = 0.3
            print('RMSE: ' + str(rmse))
            mean_points = data_trn_abs[1: n_adm + 1, :]  # fit_i[3 * n_adm:4 * n_adm][0][:len(data_in[0][1])]

            # Fix some parameters which should not be re-fit: mobility rate, steepness of R decay
            fit_alpha = result[0].x[9 * n_adm + 2]
            par_list_uncert = par_list.copy()
            par_list_uncert[9 * n_adm + 2] = fit_alpha
            bnds_uncert = list(bnds).copy()
            bnds_uncert[9 * n_adm + 2] = (fit_alpha, fit_alpha)
            bnds_uncert = tuple(bnds_uncert)

            bnds_uncert = list(bnds_uncert)
            for k in range(0, n_adm):
                fit_b = result[0].x[7 * n_adm + 1 + k]
                par_list_uncert[7 * n_adm + 1 + k] = fit_b
                bnds_uncert[7 * n_adm + 1 + k] = (fit_b, fit_b)
            bnds_uncert = tuple(bnds_uncert)

            fit_Tinf = result[0].x[9 * n_adm + 1]
            par_list_uncert[9 * n_adm + 1] = fit_Tinf
            bnds_uncert = list(bnds_uncert)
            bnds_uncert[9 * n_adm + 1] = (fit_Tinf, fit_Tinf)
            bnds_uncert = tuple(bnds_uncert)

            fit_TinfUi = result[0].x[10 * n_adm + 4]
            par_list_uncert[10 * n_adm + 4] = fit_TinfUi
            bnds_uncert = list(bnds_uncert)
            bnds_uncert[10 * n_adm + 4] = (fit_TinfUi, fit_TinfUi)
            bnds_uncert = tuple(bnds_uncert)

            # Now repeat with disturbed data for the specified number of runs - only keep runs with a minimum goodness of fit
            input = [data_trn, t_trn, model, par_list_uncert, bnds_uncert, fixed_pars, adm0, t_tst, neigh, alpha_fix,
                     inds_s10, rmse, mean_points]
            if not useparallel:
                for i in range(1, n_uncert + 1):
                    # Randomly shift data
                    out = uncertFit(input + [i])

                    data_in[i] = out[1]
                    result[i] = out[0]
                    fit[i] = out[2]

            else:

                counter = 0
                n_successruns = 0
                while n_successruns < n_uncert and counter < 10:

                    input_par = [input + [1 + counter * n_jobs]]
                    for i2 in range(2, n_jobs + 1):
                        input_par.append(input + [i2 + counter * n_jobs])

                    outs = Parallel(n_jobs=n_jobs, verbose=1, backend="multiprocessing")(
                        map(delayed(uncertFit), input_par))

                    for i2 in range(1, n_jobs + 1):
                        out = outs[i2 - 1]
                        acceptableFit = out[3]

                        if acceptableFit == 1:
                            data_in.append(out[1])
                            result.append(out[0])
                            fit.append(out[2])
                            r2s.append(out[4])
                            n_successruns = n_successruns + 1

                    print('Have ' + str(n_successruns) + ' successful runs!!!')

                    counter = counter + 1

                # finished
                if len(result) - 1 > n_uncert:
                    result = result[0:n_uncert + 1]
                    fit = fit[0:n_uncert + 1]
                    data_in = data_in[0:n_uncert + 1]

                print('TOTAL of ' + str(n_uncert) + ' suitable fits!')

            if saveDataUnc:
                if len(quat) > 1:
                    print("CANNOT SAVE DATA IN COUPLED MODE!!! - Continue without saving")
                else:
                    t_use = data_in[0][0, :]
                    data_array = np.zeros((len(data_in), len(t_use)))
                    for di in range(0, len(data_in)):
                        d = data_in[di]
                        data_array[di, :] = d[1, :]
                    df_data = pd.DataFrame(data=data_array)
                    df_data.to_csv(os.path.join(wd, 'Data', str(quat[0]) + '_data_' + uncData_id + '.csv'))
                    df_time = pd.DataFrame(data=t_use)
                    df_time.to_csv(os.path.join(wd, 'Data', str(quat[0]) + '_time_' + uncData_id + '.csv'))

    else:
        # fit model
        result_i, t_i, fit_i = fit_general(model, par_list, data_trn, bnds, fixed_pars, adm0, t_trn, t_tst,
                                           neigh, alpha_fix)
        result.append(result_i)
        fit.append(fit_i)
        print(result_i.x)

    # writing output
    core_name = '_quarter' + str(quat) + '_' + str(n_adm) + 'node' + \
                '_level-' + str(lvl) + \
                '_coupled-' + str(coupled) + \
                '_local-' + str(local) + \
                '_end-' + str(end) + \
                '_tst' + str(-n_tst) + \
                '_neigh' + str(neigh) + \
                '_afix' + str(alpha_fix) + \
                'uncert' + str(uncert)

    # parameters
    os.chdir(os.path.join(newdir, 'parameters'))
    save_pars_ILGE(adm0, core_name, result, fixed_pars, bnds, data_in)
    os.chdir(wd)

    # original and modeled data as well as figures
    save_data_general(model, adm0, quat, n_adm, core_name, fit, t_trn, t_tst, data_trn, data_tst, \
                      pop, start, end, result, fixed_pars, newdir=newdir, useUnrepor=useUnrepor)

    return 0


# Load data
def setup_ILGE(lvl, coupled, local, t_meas, n_tst, start, end, mode, quarter, model, neigh, useSoc, loadFr_dyn,
               usePangolin, pangolingStrain, mergeQuatSoc, n_splitsSoc, useUnrepor, delta_t):
    """Does the general setup based on user input.

    Parameters
    ----------
    lvl     : admin level of fitting procedure
    coupled : switches coupling on or off
    local   : switches locality of parameters rates on or off
    t_meas  : time at which interventions are modeled to takle place
    n_tst   : length of test data set
    start   : start date of analysis
    end     : end date of analysis
    mode    : include (s)ingle node only or (a)ll nodes
    qua     : list of quaters
    pop     : list of population in each quater

    Returns
    -------
    n_cmp    : number of compartments (5 in case of  S, E, I, R, D)
    A        : graph adjacency matrix
    n_adm    : number of areas to be considered
    t_trn    : vector with time points considered for training
    t_tst    : vector with time points for testing
    data_trn : data matrix to be passed to the optimization function
    data_tst : data matrix for testing the fitted model
    admo     :
    """
    # number of compartments, 5 for S, E, I, R, D

    n_adm = len(quarter)
    if model == 'SEAIRD':
        n_cmp = 8
    elif model == 'SEUI':
        if useUnrepor:
            if useReortingDelay:
                n_cmp = 7
            else:
                n_cmp = 6
        else:
            n_cmp = 4
    else:
        raise ('Invalid ODE model')

    # obtain Social adjacency matrix
    ASoc = 0
    if lvl == 0:
        ASoc = 0
    elif lvl == 1:
        if useSoc:
            if mode == 'a':  # all admin1 analyzed jointly
                if coupled:
                    ASoc = obtain_SocAdjacencyILGE(quarter)
                else:
                    ASoc = 0
            elif mode == 's':  # a single adm1 analyzed
                ASoc = 0
        else:
            ASoc = 0

    # get mobility time dependece
    if loadFr_dyn:

        # Load time searies - starts on 1.2.2020
        df_timedep = pd.read_csv(os.path.join(wd, 'output', 'bs_full_traffic_model_timeseries.csv'))
        n_travelling = df_timedep['total'].values[:-2]

        # Get the correct time frame and normalize by median number of travellers in Feb 2020
        # fraction = n_travelling[35-delta_t:]/np.median(n_travelling[:28])

        # Starting 6.2. - relative to 7.3.
        time = 7 * df_timedep.index.values[:-2] - 30 + delta_t
        # time_dep = interp1d(time, n_travelling)

        time_dep = UnivariateSpline(time, n_travelling)
        time_dep.set_smoothing_factor(0.0001)

        # Plot
        # t = np.linspace(-20, 50)
        # plt.figure()
        # plt.scatter(time, n_travelling, label = 'Data')
        # plt.plot(t,time_dep(t),label = 'Linear interpolation')
        # spl.set_smoothing_factor(0.5)
        # plt.plot(t, spl(t), label='Smooth spline 0.5')
        # spl.set_smoothing_factor(0.01)
        # plt.plot(t, spl(t), label='Smooth spline 0.01')
        # spl.set_smoothing_factor(0.005)
        # plt.plot(t, spl(t), label='Smooth spline 0.005')
        # plt.legend()

    else:
        time_dep = 0

    # Get social time series
    df_Kalman = pd.read_csv(os.path.join(wd, 'kalman', 'bs_kalman_Reff.csv'))
    time_Kalman = np.arange(0, 57)  # df_Kalman['timestamp'].values
    R_estimate = df_Kalman['R_estimate'].values

    if delta_t != 10:
        R_estimate = np.array(list(R_estimate[0] * np.ones(delta_t - 10, )) + list(R_estimate))
        time_Kalman = np.arange(0, len(R_estimate))

    alpha_mob = time_dep(time_Kalman)
    y_Kalman = R_estimate / alpha_mob
    y_soc = y_Kalman / np.max(y_Kalman)
    # y_soc = y_kalman/alpha_mob

    global time_dep_soc
    time_dep_soc = UnivariateSpline(time_Kalman, y_soc, s=0.03)

    tmin = 51
    stretch = 0.
    dates = np.array(pd.date_range(start, end))
    ts = pd.to_datetime(dates)
    dates = ts.strftime('%m.%d').values
    ticks = np.arange(min(np.arange(0, 61)), max(np.arange(0, 61)), 7)
    int_ticks = [int(i) for i in ticks]

    fig1 = plt.figure(figsize=(3, 3))
    ax = fig1.add_subplot()
    plt.plot(time_Kalman, R_estimate, label='estimated R', color='teal')
    plt.plot(time_Kalman, y_Kalman, label='R/' + r'$\alpha_{mob}$', color='darkblue')
    plt.plot(time_Kalman, y_soc, '--', label='Normalized', color='black', zorder=3)
    plt.plot(np.arange(0, 61), stretchFun(np.arange(0, 61), tmin, stretch), label='Smoothend', color='red')
    plt.ylabel('Social Interaction Score', fontsize=10)
    ax.set_xticks(ticks)
    plt.yticks(fontsize=8)
    ax.set_xticklabels(dates[int_ticks], rotation=45, fontsize=8)
    plt.legend(frameon=False, loc="upper right", fontsize=8)
    plt.tight_layout()

    fig1 = plt.figure(figsize=(3, 3))
    ax = fig1.add_subplot()
    # plt.plot(np.arange(0, 61), time_dep_soc(np.arange(0, 61)), label = '')
    plt.plot(time_Kalman, alpha_mob, label='Mobility time series', color='teal')
    plt.ylabel('Mobility Interaction Score')
    plt.yticks(fontsize=10)
    ax.set_xticks(ticks)
    ax.set_xticklabels(dates[int_ticks], rotation=45, fontsize=10)
    plt.tight_layout()

    df_mobility = pd.DataFrame(data=np.array([time_Kalman, alpha_mob]).T, index=dates, columns=['time', 'alpha_mob'])
    df_mobility.to_csv(os.path.join(wd, 'Data', 'alpha_mob.csv'))

    df_soc = pd.DataFrame(data=np.array([time_Kalman, stretchFun(np.arange(0, 61), tmin, stretch),
                                         y_soc, y_Kalman, R_estimate]).T, index=dates,
                          columns=['time', 'alpha_soc', 'R/alpha_mob norm', 'R/alpha_mob', 'R_kalman'])
    df_soc.to_csv(os.path.join(wd, 'Data', 'alpha_soc.csv'))

    # parse the input data to the right format
    t_trn, t_tst, data_trn, data_tst, pop, data_trn_abs = \
        obtain_data_ILGE(quarter, n_tst, n_adm, start, end, usePangolin, pangolingStrain, mergeQuatSoc, n_splitsSoc)

    # obtain adjacency matrix and number of admin areas analyzed
    A = 0
    if lvl == 0:
        A = 0
    elif lvl == 1:
        if mode == 'a':  # all admin1 analyzed jointly
            if coupled:

                if useSymMat:
                    A = obtain_adjacencyILGE(quarter, neigh, n_splitsSoc)
                else:
                    A = obtain_adjacencyILGE_NormRow(quarter, neigh, n_splitsSoc, pop)
            else:
                A = 0
        elif mode == 's':  # a single adm1 analyzed
            A = 0

    plt.close('all')

    return n_cmp, A, n_adm, t_trn, t_tst, data_trn, data_tst, pop, ASoc, time_dep, data_trn_abs


def obtain_data_ILGE(quarter, n_tst, n_adm, start, end, usePangolin, pangolingStrain, mergeQuatSoc, n_splitsSoc):
    """Obtains the dataset in the right format to be passed to optimizer.

    Parameters
    ----------
    quarter : list of quaters
    n_tst : negative integer which specifies length of test data set
    n_adm : number of considered admin areas
    start : start date of analysis
    end   : end date of analysis

    Returns
    -------
    t_trn    : vector with time points considered for training
    t_tst    : vector with time points for testing
    data_trn : data matrix to be passed to the optimization function
    data_tst : data to be reserved to test the fitted model
    pop      : population per each area
    """

    # allocation
    data_inf_trn = np.array([])
    data_inf_trn_cum = np.array([])
    data_fat_trn = np.array([])
    data_inf_tst = np.array([])
    data_inf_tst_cum = np.array([])
    data_fat_tst = np.array([])
    measures = []
    pop = []

    for j, c in enumerate(quarter):
        print(c)

        if localMac:
            data, subpop = load_local_data_ILGElocal(c, start, end, usePangolin, pangolingStrain, mergeQuatSoc,
                                                     n_splitsSoc)
        else:
            data, subpop, baselstrain, sequenced, posCases, t_poscases = load_local_data_ILGE(c, start, end,
                                                                                              usePangolin,
                                                                                              pangolingStrain,
                                                                                              mergeQuatSoc, n_splitsSoc)

        if j == 0:
            if (n_tst > data[0][-1]):
                print('\nTest set larger than total dataset!\n')
            t_trn = data[0][:n_tst]
            t_tst = data[0][n_tst:]

        # measures.append(t_meas)# for now we use a global measure
        pop.append(subpop)

        # data (1st column is time, the rest are dead and infected, in couples)
        data_inf_trn_cum = np.concatenate((data_inf_trn_cum, data[2][:n_tst]))
        data_inf_tst_cum = np.concatenate((data_inf_tst_cum, data[2][n_tst:]))
        data_inf_trn = np.concatenate((data_inf_trn, data[3][:n_tst]))
        data_inf_tst = np.concatenate((data_inf_tst, data[3][n_tst:]))
        # data_fat_tst = np.concatenate((data_fat_tst, data[1][n_tst:]))
        # data_fat_trn = np.concatenate((data_fat_trn, data[1][:n_tst]))

    # training data: first row is time, then cummulative number of cases
    inf_trn = data_inf_trn.reshape(n_adm, len(t_trn))
    inf_trn_cum = data_inf_trn_cum.reshape(n_adm, len(t_trn))
    # fat_trn = data_fat_trn.reshape(n_adm, len(t_trn))

    # test data
    inf_tst = data_inf_tst.reshape(n_adm, len(t_tst))
    # fat_tst = data_fat_tst.reshape(n_adm, len(t_tst))

    # the actual dataset used for the fit: each column is an adm area.
    # data_trn = np.concatenate((t_trn[None, :], inf_trn, fat_trn), axis=0)
    # data_tst = np.concatenate((t_tst[None, :], inf_tst, fat_tst), axis=0)

    # Impute missing data for the 29, 30, 31 of march
    inds_missing = [36, 37, 38]
    inds_forRatio = [33, 34, 35, 19, 40, 41]
    ratioT = (inf_trn_cum / np.sum(inf_trn_cum, axis=0))[:, inds_forRatio].mean(axis=1)
    ind36 = np.where(t_poscases == 36)[0][0]
    cases = sequenced * baselstrain * posCases[ind36:ind36 + 3]
    inf_trn_imp = inf_trn.copy()
    for counter, i_miss in enumerate(inds_missing):
        for q_rep in range(0, n_adm):
            inf_trn_imp[q_rep, i_miss] = cases[counter] * ratioT[q_rep]

    # 7 day average
    if use7DayAV:
        inf_trn_cum7av = np.zeros(inf_trn_imp.shape)
        inf_trn_7av = np.zeros(inf_trn_imp.shape)
        inf_trn_cum7av_noImp = np.zeros(inf_trn_imp.shape)
        inf_trn_7av_noImp = np.zeros(inf_trn_imp.shape)
        for q in range(0, n_adm):
            this_inf_trn = inf_trn_imp[q, :]
            this_inf_trn_noImp = inf_trn[q, :]
            y_inf_new_7av = np.zeros(this_inf_trn.shape)
            y_inf_new_7av_noImp = np.zeros(this_inf_trn.shape)
            for d in range(0, len(this_inf_trn)):
                ind = np.arange(d - 3, d + 4)
                ind_use = ind[np.logical_and(ind >= 0, ind < len(this_inf_trn))]
                y_inf_new_7av[d] = np.mean(np.array(this_inf_trn)[ind_use])
                y_inf_new_7av_noImp[d] = np.mean(np.array(this_inf_trn_noImp)[ind_use])
            y_inf_7av = np.cumsum(y_inf_new_7av)
            y_inf_7av_noImp = np.cumsum(y_inf_new_7av_noImp)

            inf_trn_cum7av[q, :] = y_inf_7av
            inf_trn_7av[q, :] = y_inf_new_7av
            inf_trn_cum7av_noImp[q, :] = y_inf_7av_noImp
            inf_trn_7av_noImp[q, :] = y_inf_new_7av_noImp

    # Plot
    dates = np.array(pd.date_range(start, end))
    ts = pd.to_datetime(dates)
    dates = ts.strftime('%d %b').values
    t = list(t_trn) + list(t_tst)
    ticks = np.arange(min(t_trn) + 2, max(t), 7)
    int_ticks = [int(i) for i in ticks]

    cmap = cmx.get_cmap("Spectral")
    fig1 = plt.figure(figsize=(6, 6))
    ax = fig1.add_subplot(1, 1, 1)
    plt.plot(t_trn, inf_trn_7av[0, :], color='seagreen', label='T1 imputed')
    plt.plot(t_trn, inf_trn_7av[1, :], color='crimson', label='T2 imputed')
    plt.plot(t_trn, inf_trn_7av[2, :], color='blue', label='T3 imputed')
    plt.plot(t_trn, inf_trn_7av_noImp[0, :], '--', color='seagreen', label='T1')
    plt.plot(t_trn, inf_trn_7av_noImp[1, :], '--', color='crimson', label='T2')
    plt.plot(t_trn, inf_trn_7av_noImp[2, :], '--', color='blue', label='T3')
    ax.set_xticks(ticks)
    plt.yticks(fontsize=16)
    ax.set_xticklabels(dates[int_ticks], rotation=45, fontsize=16)
    plt.legend(prop={'size': 14}, frameon=False, loc="upper left")
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.ylabel('Number of Cases', fontsize=16)
    ax.patch.set_facecolor('lightgrey')
    plt.grid(color='white')
    plt.tight_layout()

    if useImputation:
        data_trn = np.concatenate((t_trn[None, :], inf_trn_cum7av), axis=0)
        data_tst = np.concatenate((t_tst[None, :], inf_tst), axis=0)
        data_trn_abs = np.concatenate((t_trn[None, :], inf_trn_7av), axis=0)

        # Save
        time_imp = data_trn[0, :]
        count_abs1 = data_trn_abs[1, :]
        count_abs2 = data_trn_abs[2, :]
        count_abs3 = data_trn_abs[3, :]
        count_cum1 = data_trn[1, :]
        count_cum2 = data_trn[2, :]
        count_cum3 = data_trn[3, :]

        df_dataTrain = pd.DataFrame(
            list(zip(time_imp, count_abs1, count_abs2, count_abs3, count_cum1, count_cum2, count_cum3)),
            columns=['time', 'count_abs1', 'count_abs2', 'count_abs3', 'count_cum1', 'count_cum2', 'count_cum3'])

        df_dataTrain.to_csv(os.path.join(wd, 'Data', 'imputedBaselStrain.csv'))

        df_liv = pd.read_csv(os.path.join(wd, 'Data', 'imputedBaselStrainLivingSpace.csv'))
        df_sen = pd.read_csv(os.path.join(wd, 'Data', 'imputedBaselStrain1PHouseholds.csv'))
        df_1P = pd.read_csv(os.path.join(wd, 'Data', 'imputedBaselStrainSENIOR_ANT.csv'))
        df_med = pd.read_csv(os.path.join(wd, 'Data', 'imputedBaselStrainMedianIncome2017.csv'))

        # Read all data and compare
        df_allPartitions = pd.DataFrame(df_dataTrain['time'].copy())
        df_allPartitions['MedInc_abs'] = df_med['count_abs1'] + df_med['count_abs2'] + df_med['count_abs3']
        df_allPartitions['Liv_abs'] = df_liv['count_abs1'] + df_liv['count_abs2'] + df_liv['count_abs3']
        df_allPartitions['1P_abs'] = df_1P['count_abs1'] + df_1P['count_abs2'] + df_1P['count_abs3']
        df_allPartitions['Sen_abs'] = df_sen['count_abs1'] + df_sen['count_abs2'] + df_sen['count_abs3']

        df_allPartitions['MedInc_cum'] = df_med['count_cum1'] + df_med['count_cum2'] + df_med['count_cum3']
        df_allPartitions['Liv_cum'] = df_liv['count_cum1'] + df_liv['count_cum2'] + df_liv['count_cum3']
        df_allPartitions['1P_cum'] = df_1P['count_cum1'] + df_1P['count_cum2'] + df_1P['count_cum3']
        df_allPartitions['Sen_cum'] = df_sen['count_cum1'] + df_sen['count_cum2'] + df_sen['count_cum3']
        df_allPartitions.to_csv(os.path.join(wd, 'Data', 'imputedBaselStrain_All.csv'))






    else:
        data_trn = np.concatenate((t_trn[None, :], inf_trn_cum7av_noImp), axis=0)
        data_tst = np.concatenate((t_tst[None, :], inf_tst), axis=0)
        data_trn_abs = np.concatenate((t_trn[None, :], inf_trn_7av_noImp), axis=0)

    return t_trn, t_tst, data_trn, data_tst, pop, data_trn_abs


def load_local_data_ILGElocal(the_quarter, start, end, usePangolin, pangolingStrain, mergeQuatSoc, n_splitsSoc):
    '''
    Parameters
    ----------
    the_quarter  : the areas of interest, may be 'all'
    start      : start date of analysis
    end        : end date of analysis

    Returns
    --------
    data matrix with time, n_fats, infected; and population for selected areas
    '''

    if mergeQuatSoc:

        # read csv files for cases
        # os.chdir('/project/data/20200722T103018/content/')#20200528T091303/content/epiData')
        os.chdir('/Users/sbrueningk/Desktop/ILGE_Data/')
        filename = os.path.join(wd, 'Data',
                                'data.csv')  # 'EpiData_ETH_complete_20200720_extern.xlsx' #'EpiData_ETH_censor_20200515.xlsx'
        # df = pd.read_excel(filename, 'positive')
        df = pd.read_csv(filename)
        # df_neg = pd.read_excel(filename, 'negative')
        os.chdir(wd)

        # fill in data for Riehen and Bettingen
        # df.Quarters.fillna(df.Ortschaft, inplace = True)

        # restrict to dates greater than starting date
        df['ENTNAHMEDATUM'] = pd.to_datetime(df['ENTNAHMEDATUM'], format='%Y-%m-%d')
        df['ENTNAHMEDATUM'] = df['ENTNAHMEDATUM'].dt.date
        START = df['ENTNAHMEDATUM'] >= start
        END = df['ENTNAHMEDATUM'] <= end
        df = df[START & END]
        tmp = (df['ENTNAHMEDATUM'] - start)
        df['DELTA'] = tmp.astype('timedelta64[D]')
        t = np.arange(df['DELTA'].max() + 1)  # np.array(np.unique(df['DELTA']))

        # # Get socioeconomic data
        os.chdir(os.path.join(wd, 'geodata'))
        filename = 'SocioeconomicScore_data.csv'
        pop_df = pd.read_csv(filename)
        os.chdir(wd)

        if useNaturalBreaks:
            name_suffix = 'natural_breaks'
        else:
            name_suffix = 'percentiles'
        os.chdir(os.path.join(wd, 'graphs'))
        if useForTiles == 'MedianIncome2017':
            filename = 'bs_MedianIncome2017_' + str(n_splitsSoc) + name_suffix + '.csv'
        elif useForTiles == 'CoHab_index':
            filename = 'bs_CohabProxIndex_' + str(n_splitsSoc) + name_suffix + '.csv'
        elif useForTiles == 'SENIOR_ANT':
            filename = 'bs_SENIOR_ANT_' + str(n_splitsSoc) + name_suffix + '.csv'
        elif useForTiles == 'LivingSpace':
            filename = 'bs_Living_space_per_Person_2017_' + str(n_splitsSoc) + name_suffix + '.csv'
        elif useForTiles == '1PHouseholds':
            filename = 'bs_1PHouseholds_' + str(n_splitsSoc) + name_suffix + '.csv'
        elif useForTiles == 'random':
            filename = 'bs_random_' + str(n_splitsSoc) + 'tiles_' + randomIndex + '.csv'
        else:
            filename = 'bs_' + useForTiles + '_' + str(n_splitsSoc) + name_suffix + '.csv'
        soc_df = pd.read_csv(filename)
        os.chdir(wd)

        if the_quarter == 'all':
            raise ('The option all is not supported for splitting according to living space')
        else:

            # Get total number of inhabitants
            os.chdir(os.path.join(wd, 'geodata'))
            filename = 'bs_quarter_mapping_all.csv'
            pop_tot_df = pd.read_csv(filename)
            os.chdir(wd)

            # For each tile get the blocks and population
            blocks = soc_df['BLO_ID'].loc[soc_df['percentile'] == the_quarter].values

            # Split according to living space
            pop_df_nan = pop_df.dropna(axis=0)
            if the_quarter == 0:

                # # Use NaNs
                # blocks = pop_df['BlockID'].loc[pop_df['Living space per Person 2017'].isnull()].values
                # pop = pop_tot_df['POPULATION'].sum() - pop_df_nan['Population 2017'].sum()

                otherBlocks = soc_df['BLO_ID'].loc[soc_df['percentile'] != the_quarter].values
                pop = pop_tot_df['POPULATION'].sum() - pop_df['Population 2017'].loc[
                    pop_df['BlockID'].isin(otherBlocks)].sum()
            else:
                # if the_quarter == n_splitsSoc:
                #     cutoff_up = max(pop_df['Living space per Person 2017'])
                # else:
                #     cutoff_up = np.percentile(pop_df_nan['Living space per Person 2017'], 100 / n_splitsSoc * the_quarter)
                #
                # if the_quarter>1:
                #     cutoff_low = np.percentile(pop_df_nan['Living space per Person 2017'], 100 / n_splitsSoc * (the_quarter-1))
                # else:
                #     cutoff_low = 0.9*np.percentile(pop_df_nan['Living space per Person 2017'],0)
                #
                #
                # # Get the blocks belonging to this living space percentile
                # blocks = pop_df['BlockID'].loc[(cutoff_low < pop_df['Living space per Person 2017']) & (pop_df['Living space per Person 2017']<=cutoff_up)].values

                # Get population
                pop = pop_df['Population 2017'].loc[pop_df['BlockID'].isin(blocks)].sum()

            try:
                # Subset the case data
                print(list(blocks))
                # df = df#[df['BlockID'].isin(list(blocks))]
                print(str(the_quarter) + " has a total of " + str(df.shape[0]) + ' cases')
            except:
                raise ('Subset for quarter failed!!!')

        # Produce time series
        counts_del = df['DELTA'].value_counts().values
        labels_del = df['DELTA'].value_counts().keys().values.astype('timedelta64[D]')
        y_new_inf = [x for _, x in sorted(zip(labels_del, counts_del))]
        tsub = np.sort(labels_del)
        tsub = tsub.astype('timedelta64[D]') / np.timedelta64(1, 'D')
        for k in range(0, len(t)):
            if t[k] not in tsub:
                y_new_inf.insert(k, 0)

        y_inf = np.cumsum(y_new_inf)

        # Subset age group (not used at this stage yet)
        sorted_counts1, sorted_labels1 = getAGE_timeSeries(df, 1)
        sorted_counts2, sorted_labels2 = getAGE_timeSeries(df, 2)
        sorted_counts3, sorted_labels3 = getAGE_timeSeries(df, 3)
        sorted_counts4, sorted_labels4 = getAGE_timeSeries(df, 4)

        # So far no data for fatalities
        y_dead = []

        # Use 7-day average
        if use7DayAV:
            y_inf_7av = np.zeros(y_inf.shape)
            y_inf_new_7av = np.zeros(y_inf.shape)
            for d in range(0, len(y_inf)):
                ind = np.arange(d - 3, d + 4)
                ind_use = ind[np.logical_and(ind >= 0, ind < len(y_inf))]
                y_inf_new_7av[d] = np.mean(np.array(y_new_inf)[ind_use])
            y_inf_7av = np.cumsum(y_inf_new_7av)

            y_inf = y_inf_7av
            y_new_inf = y_inf_new_7av

        # Summarize
        data = []
        data.append(t)
        data.append(y_dead)
        data.append(y_inf)
        data.append(y_new_inf)
    else:

        # read csv files for cases
        # os.chdir('/project/data/20200722T103018/content/')#20200528T091303/content/epiData')
        os.chdir('/Users/sbrueningk/Desktop/ILGE_Data/')
        filename = os.path.join(wd, 'Data',
                                'data.csv')  # 'EpiData_ETH_complete_20200720_extern.xlsx' #'EpiData_ETH_censor_20200515.xlsx'
        # df = pd.read_excel(filename, 'positive')
        df = pd.read_csv(filename)
        # df_neg = pd.read_excel(filename, 'negative')
        os.chdir(wd)

        # fill in data for Riehen and Bettingen
        # df.Quarters.fillna(df.Ortschaft, inplace = True)

        # read soc'ioeconomic data
        os.chdir(os.path.join(wd, 'geodata'))
        # filename = 'SocioeconomicScore_data.xls'
        filename = 'bs_quarter_mapping_all.csv'
        pop_df = pd.read_csv(filename)
        os.chdir(wd)

        # restrict to dates greater than starting date
        df['ENTNAHMEDATUM'] = pd.to_datetime(df['ENTNAHMEDATUM'], format='%Y-%m-%d')
        df['ENTNAHMEDATUM'] = df['ENTNAHMEDATUM'].dt.date
        START = df['ENTNAHMEDATUM'] >= start
        END = df['ENTNAHMEDATUM'] <= end
        df = df[START & END]
        tmp = (df['ENTNAHMEDATUM'] - start)
        df['DELTA'] = tmp.astype('timedelta64[D]')
        t = np.arange(df['DELTA'].max() + 1)  # np.array(np.unique(df['DELTA']))

        # Subset for selected adm1_pcode
        if the_quarter == 'all':
            pop = pop_df['POPULATION'].sum()
        else:

            # Get the quarters beloning to this ID
            quarters = pop_df['GEBIET'].loc[pop_df['ID_PARTNER'] == the_quarter].values
            print('Using these quarters:', quarters)

            # Get population
            pop = pop_df['POPULATION'].loc[pop_df['ID_PARTNER'] == the_quarter].sum()

            try:
                # Subset the case data
                df = df[df['Block ID'].isin(list(quarters))]
                print(quarters + " has a total of " + str(df.shape[0]) + ' cases')
            except:
                raise ('Subset for quarter failed!!!')

        # Produce time series
        counts_del = df['DELTA'].value_counts().values
        labels_del = df['DELTA'].value_counts().keys().values.astype('timedelta64[D]')
        y_new_inf = [x for _, x in sorted(zip(labels_del, counts_del))]
        tsub = np.sort(labels_del)
        tsub = tsub.astype('timedelta64[D]') / np.timedelta64(1, 'D')
        for k in range(0, len(t)):
            if t[k] not in tsub:
                y_new_inf.insert(k, 0)

        y_inf = np.cumsum(y_new_inf)

        # Subset age group (not used at this stage yet)
        sorted_counts1, sorted_labels1 = getAGE_timeSeries(df, 1)
        sorted_counts2, sorted_labels2 = getAGE_timeSeries(df, 2)
        sorted_counts3, sorted_labels3 = getAGE_timeSeries(df, 3)
        sorted_counts4, sorted_labels4 = getAGE_timeSeries(df, 4)

        # So far no data for fatalities
        y_dead = []

        # Summarize
        data = []
        data.append(t)
        data.append(y_dead)
        data.append(y_inf)
        data.append(y_new_inf)

    return data, pop


def load_local_data_ILGE(the_quarter, start, end, usePangolin, pangolingStrain, mergeQuatSoc, n_splitsSoc):
    '''
    Parameters
    ----------
    the_quarter  : the areas of interest, may be 'all'
    start      : start date of analysis
    end        : end date of analysis

    Returns
    --------
    data matrix with time, n_fats, infected; and population for selected areas
    '''
    global randomIndex

    if mergeQuatSoc:

        if usePangolin:

            # read csv files for cases
            os.chdir('Data')

            # os.chdir('/project/data/20200817T141323/content/')
            filename = 'EpiData_ETH_complete_20201104_final.xlsx'
            df = pd.read_excel(filename, 'positive')
            # df_neg = pd.read_excel(filename, 'negative')
            os.chdir(wd)

            # drop some cases
            df = df.drop(df[df['AUFTRAGSNUMMER'] == 42184023].index.values[0])
            df = df.drop(df[df['AUFTRAGSNUMMER'] == 42186104].index.values[0])
            df = df.drop(df[df['AUFTRAGSNUMMER'] == 42189099].index.values[0])
            df = df.drop(df[df['AUFTRAGSNUMMER'] == 42189999].index.values[0])
            df = df.drop(df[df['AUFTRAGSNUMMER'] == 42206243].index.values[0])
            df = df.drop(df[df['AUFTRAGSNUMMER'] == 42206243].index.values[0])

            # read csv files for cases
            os.chdir('Data')
            df_pos = pd.read_excel(filename, 'positive')
            df_neg = pd.read_excel(filename, 'negative')
            os.chdir(wd)
            df_pos = df_pos.drop(df_pos[df_pos['AUFTRAGSNUMMER'] == 42184023].index.values[0])
            df_pos = df_pos.drop(df_pos[df_pos['AUFTRAGSNUMMER'] == 42186104].index.values[0])
            df_pos = df_pos.drop(df_pos[df_pos['AUFTRAGSNUMMER'] == 42189099].index.values[0])
            df_pos = df_pos.drop(df_pos[df_pos['AUFTRAGSNUMMER'] == 42189999].index.values[0])
            df_pos = df_pos.drop(df_pos[df_pos['AUFTRAGSNUMMER'] == 42206243].index.values[0])

            df_pos.drop_duplicates(subset=['AUFTRAGSNUMMER'])
            df_pos['ENTNAHMEDATUM'] = pd.to_datetime(df_pos['ENTNAHMEDATUM'], format='%Y-%m-%d')
            df_pos['ENTNAHMEDATUM'] = df_pos['ENTNAHMEDATUM'].dt.date
            START = df_pos['ENTNAHMEDATUM'] >= start
            END = df_pos['ENTNAHMEDATUM'] <= end
            df_pos = df_pos[START & END]
            tmp = (df_pos['ENTNAHMEDATUM'] - start)
            df_pos['DELTA'] = tmp.astype('timedelta64[D]')
            time_pos = df_pos['DELTA'].value_counts().index
            case_pos = df_pos['DELTA'].value_counts().values

            df_neg.drop_duplicates(subset=['AUFTRAGSNUMMER'])
            df_neg['ENTNAHMEDATUM'] = pd.to_datetime(df_neg['ENTNAHMEDATUM'], format='%Y-%m-%d')
            df_neg['ENTNAHMEDATUM'] = df_neg['ENTNAHMEDATUM'].dt.date
            START = df_neg['ENTNAHMEDATUM'] >= start
            END = df_neg['ENTNAHMEDATUM'] <= end
            df_neg = df_neg[START & END]
            tmp = (df_neg['ENTNAHMEDATUM'] - start)
            df_neg['DELTA'] = tmp.astype('timedelta64[D]')
            time_neg = df_neg['DELTA'].value_counts().index
            case_neg = df_neg['DELTA'].value_counts().values

            # interpolate
            t_posneg = np.arange(min(time_pos), max(time_pos) + 1)
            interpPos = interp1d(time_pos, case_pos)
            interpNeg = interp1d(time_neg, case_neg)
            case_posInt = interpPos(t_posneg)
            case_negInt = interpNeg(t_posneg)

            # get 7 day moving average
            if use7DayAV:
                case_pos_7av = np.zeros(case_posInt.shape)
                for d in range(0, len(case_posInt)):
                    ind = np.arange(d - 3, d + 4)
                    ind_use = ind[np.logical_and(ind >= 0, ind < len(case_posInt))]
                    case_pos_7av[d] = np.mean(np.array(case_posInt)[ind_use])
                posCases = case_pos_7av
            else:
                posCases = case_posInt
            t_poscases = t_posneg

            # Get ratio
            ratio = case_posInt / (case_negInt + case_posInt)
            ratio_7av = np.zeros(ratio.shape)
            for d in range(0, len(ratio)):
                ind = np.arange(d - 3, d + 4)
                ind_use = ind[np.logical_and(ind >= 0, ind < len(ratio))]
                ratio_7av[d] = np.mean(ratio[ind_use])
            ratio_use = ratio_7av  # /np.mean(ratio_7av)
            ratioPosNeg_time_dep = UnivariateSpline(t_posneg, ratio_use, s=0.0001)

            fig1 = plt.figure(figsize=(6, 6))
            ax = fig1.add_subplot()
            plt.plot(t_posneg, case_posInt / (case_negInt + case_posInt), '-', linewidth=2, label='Ratio p/(p+n)')
            plt.plot(t_posneg, ratio_7av, '-', linewidth=2, label='7d av Ratio')
            plt.plot(t_posneg, ratioPosNeg_time_dep(t_posneg), '-', linewidth=2,
                     label='Smoothened Interpolation used')
            plt.xlabel('Day since 26.2.2020')
            plt.ylabel('Interaction Fraction')
            plt.legend()

            fig1 = plt.figure(figsize=(6, 6))
            ax = fig1.add_subplot()
            plt.plot(time_neg, case_neg, '-', linewidth=2, label='Negative')
            plt.plot(time_pos, case_pos, '-', linewidth=2, label='Positive')
            # plt.plot(t_posneg, case_posInt/case_negInt, '-', linewidth=2, label='Ratio')
            plt.xlabel('Day since 26.2.2020')
            plt.ylabel('Interaction Fraction')
            plt.legend()

            ratioPosNeg_time_dep_norm = UnivariateSpline(t_posneg, 0.999 / 0.85 * ratio_use / max(ratio_use), s=0.0001)
            ratioPosNeg_time_dep_norm2 = UnivariateSpline(t_posneg, (0.999 / 0.85 - max(ratio_use)) + ratio_use,
                                                          s=0.0001)

            # fig1 = plt.figure(figsize=(6, 6))
            # ax = fig1.add_subplot()
            # plt.plot(t_posneg, ratio_use, '-', linewidth=2, label='7d av Ratio')
            # plt.plot(t_posneg, stretchFunPosNeg(t_posneg,tmin,0, ratioPosNeg_time_dep_norm), '-', linewidth=2, label='7d av Ratio')
            # plt.plot(t_posneg, stretchFunPosNeg(t_posneg, tmin, 0.3, ratioPosNeg_time_dep_norm), '-', linewidth=2,
            #          label='7d av Ratio')
            # plt.plot(t_posneg, stretchFunPosNeg2(t_posneg, tmin, 0., ratioPosNeg_time_dep_norm), '-', linewidth=2,
            #          label='7d av Ratio')
            # plt.xlabel('Day since 26.2.2020')
            # plt.ylabel('Interaction Fraction')
            # plt.legend()

            # Get Basel strain fraction over time
            df_notBS = df[df['orf1b_C15324T'] == 'no']
            df_notBS = df_notBS.drop_duplicates(subset=['AUFTRAGSNUMMER'])
            df_notBS['ENTNAHMEDATUM'] = pd.to_datetime(df_notBS['ENTNAHMEDATUM'], format='%Y-%m-%d')
            df_notBS['ENTNAHMEDATUM'] = df_notBS['ENTNAHMEDATUM'].dt.date
            START = df_notBS['ENTNAHMEDATUM'] >= start
            END = df_notBS['ENTNAHMEDATUM'] <= end
            df_notBS = df_notBS[START & END]
            tmp = (df_notBS['ENTNAHMEDATUM'] - start)
            df_notBS['DELTA'] = tmp.astype('timedelta64[D]')
            time_notBS = np.array(np.unique(df_notBS['DELTA']))
            case_notBS = df_notBS['DELTA'].value_counts().values

            df_BS = df[df['orf1b_C15324T'] == 'yes']
            df_BS = df_BS.drop_duplicates(subset=['AUFTRAGSNUMMER'])
            df_BS['ENTNAHMEDATUM'] = pd.to_datetime(df_BS['ENTNAHMEDATUM'], format='%Y-%m-%d')
            df_BS['ENTNAHMEDATUM'] = df_BS['ENTNAHMEDATUM'].dt.date
            START = df_BS['ENTNAHMEDATUM'] >= start
            END = df_BS['ENTNAHMEDATUM'] <= end
            df_BS = df_BS[START & END]
            tmp = (df_BS['ENTNAHMEDATUM'] - start)
            df_BS['DELTA'] = tmp.astype('timedelta64[D]')

            time_BS = np.array(np.unique(df_BS['DELTA']))
            case_BS = df_BS['DELTA'].value_counts().values

            t_BSnotBS = np.arange(min([min(time_BS), min(time_notBS)]), max([max(time_BS), max(time_notBS)]) + 1)
            interpBS = interp1d(time_BS, case_BS)
            interpnotBS = interp1d(time_notBS, case_notBS)
            case_BSInt = np.zeros(t_BSnotBS.shape)
            for t in range(0, len(t_BSnotBS)):
                if t_BSnotBS[t] >= min(time_BS) and t_BSnotBS[t] <= max(time_BS):
                    case_BSInt[t] = interpBS(t_BSnotBS[t])

            case_notBSInt = np.ones(t_BSnotBS.shape)
            for t in range(0, len(t_BSnotBS)):
                if t_BSnotBS[t] >= min(time_notBS) and t_BSnotBS[t] <= max(time_notBS):
                    case_notBSInt[t] = interpnotBS(t_BSnotBS[t])

            ratioBS = case_BSInt / (case_notBSInt + case_BSInt)
            ratioBS_7av = np.zeros(ratioBS.shape)
            for d in range(0, len(ratioBS)):
                ind = np.arange(d - 3, d + 4)
                ind_use = ind[np.logical_and(ind >= 0, ind < len(ratioBS))]
                ratioBS_7av[d] = np.mean(ratioBS[ind_use])
            ratioBS_use = ratioBS_7av / np.max(ratioBS_7av)
            baselstrain = np.mean(ratioBS_7av[15:])
            sequenced = (df_notBS.shape[0] + df_BS.shape[0]) / df_pos.shape[0]
            global ratioBS_time_dep
            ratioBS_time_dep = UnivariateSpline(t_BSnotBS, ratioBS_use, s=0.0001)

            fig1 = plt.figure(figsize=(6, 6))
            ax = fig1.add_subplot()
            plt.plot(t_BSnotBS, ratioBS_7av, '-', linewidth=2, label='7day av Basel strain percentage')
            plt.xlabel('Day since 26.2.2020')
            plt.ylabel('Interaction Fraction')
            plt.legend()

            fig1 = plt.figure(figsize=(6, 6))
            ax = fig1.add_subplot()
            plt.plot(t_BSnotBS, ratioPosNeg_time_dep(t_BSnotBS), '-', linewidth=2, label='norm 7d av Ratio BS*RatioPos')
            plt.xlabel('Day since 26.2.2020')
            plt.ylabel('Interaction Fraction')
            plt.legend()

            g = ratioBS_use * ratioPosNeg_time_dep(t_BSnotBS)
            p = ratioPosNeg_time_dep(t_posneg)
            u = findFct(11.94, p)
            global ratio_time_dep
            # ratio_time_dep = UnivariateSpline(t_BSnotBS, u)

            # Subset for the pangolin strain
            df = df[df['orf1b_C15324T'] == pangolingStrain]
            df = df.drop_duplicates(subset=['AUFTRAGSNUMMER'])

            # fill in data for Riehen and Bettingen
            df.Quarters.fillna(df.ORTSCHAFT, inplace=True)

            # restrict to dates greater than starting date
            df['ENTNAHMEDATUM'] = pd.to_datetime(df['ENTNAHMEDATUM'], format='%Y-%m-%d')
            df['ENTNAHMEDATUM'] = df['ENTNAHMEDATUM'].dt.date
            START = df['ENTNAHMEDATUM'] >= start
            END = df['ENTNAHMEDATUM'] <= end
            df = df[START & END]
            tmp = (df['ENTNAHMEDATUM'] - start)
            df['DELTA'] = tmp.astype('timedelta64[D]')
            t = np.arange(df['DELTA'].max() + 5)  # np.array(np.unique(df['DELTA']))

            # Get all population data
            os.chdir(os.path.join(wd, 'geodata'))
            filename = 'SocioeconomicScore_data.csv'
            pop_df = pd.read_csv(filename)
            os.chdir(wd)

            # Get socioeconomic data
            if useNaturalBreaks:
                name_suffix = 'natural_breaks'
            else:
                name_suffix = 'percentiles'
            os.chdir(os.path.join(wd, 'graphs'))
            if useForTiles == 'MedianIncome2017':
                filename = 'bs_MedianIncome2017_' + str(n_splitsSoc) + name_suffix + '.csv'
            elif useForTiles == 'CoHab_index':
                filename = 'bs_CohabProxIndex_' + str(n_splitsSoc) + name_suffix + '.csv'
            elif useForTiles == 'SENIOR_ANT':
                filename = 'bs_SENIOR_ANT_' + str(n_splitsSoc) + name_suffix + '.csv'
            elif useForTiles == 'LivingSpace':
                filename = 'bs_Living_space_per_Person_2017_' + str(n_splitsSoc) + name_suffix + '.csv'
            elif useForTiles == '1PHouseholds':
                filename = 'bs_1PHouseholds_' + str(n_splitsSoc) + name_suffix + '.csv'
            elif useForTiles == 'random':
                filename = 'bs_random_' + str(n_splitsSoc) + 'tiles_' + randomIndex + '.csv'
            else:
                filename = 'bs_' + useForTiles + '_' + str(n_splitsSoc) + name_suffix + '.csv'
                # raise ('Invalid choice of separation for tiles')
            soc_df = pd.read_csv(filename)
            os.chdir(wd)

            if the_quarter == 'all':
                raise ('The option all is not supported for splitting according to living space')
            else:

                # Get total number of inhabitants
                os.chdir(os.path.join(wd, 'geodata'))
                filename = 'bs_quarter_mapping_all.csv'
                pop_tot_df = pd.read_csv(filename)
                os.chdir(wd)

                # For each tile get the blocks and population
                blocks = soc_df['BLO_ID'].loc[soc_df['percentile'] == the_quarter].values

                # Split according to living space
                pop_df_nan = pop_df.dropna(axis=0)
                if the_quarter == 0:

                    # # Use NaNs
                    # blocks = pop_df['BlockID'].loc[pop_df['Living space per Person 2017'].isnull()].values
                    # pop = pop_tot_df['POPULATION'].sum() - pop_df_nan['Population 2017'].sum()

                    otherBlocks = soc_df['BLO_ID'].loc[soc_df['percentile'] != the_quarter].values
                    pop = pop_tot_df['POPULATION'].sum() - pop_df['Population 2017'].loc[
                        pop_df['BlockID'].isin(otherBlocks)].sum()
                else:
                    # if the_quarter == n_splitsSoc:
                    #     cutoff_up = max(pop_df['Living space per Person 2017'])
                    # else:
                    #     cutoff_up = np.percentile(pop_df_nan['Living space per Person 2017'], 100 / n_splitsSoc * the_quarter)
                    #
                    # if the_quarter>1:
                    #     cutoff_low = np.percentile(pop_df_nan['Living space per Person 2017'], 100 / n_splitsSoc * (the_quarter-1))
                    # else:
                    #     cutoff_low = 0.9*np.percentile(pop_df_nan['Living space per Person 2017'],0)
                    #
                    #
                    # # Get the blocks belonging to this living space percentile
                    # blocks = pop_df['BlockID'].loc[(cutoff_low < pop_df['Living space per Person 2017']) & (pop_df['Living space per Person 2017']<=cutoff_up)].values

                    # Get population
                    pop = pop_df['Population 2017'].loc[pop_df['BlockID'].isin(blocks)].sum()

                try:
                    # Subset the case data
                    df = df[df['Block ID'].isin(list(blocks))]
                    print(str(the_quarter) + " has a total of " + str(df.shape[0]) + ' cases')
                except:
                    raise ('Subset for quarter failed!!!')

            counts_del = df['DELTA'].value_counts().values
            labels_del = df['DELTA'].value_counts().keys().values.astype('timedelta64[D]')
            y_new_inf = [x for _, x in sorted(zip(labels_del, counts_del))]
            tsub = np.sort(labels_del)
            tsub = tsub.astype('timedelta64[D]') / np.timedelta64(1, 'D')
            for k in range(0, len(t)):
                if t[k] not in tsub:
                    y_new_inf.insert(k, 0)

            y_inf = np.cumsum(y_new_inf)

            # Subset age group (not used at this stage yet)
            sorted_counts1, sorted_labels1 = getAGE_timeSeries(df, 1)
            sorted_counts2, sorted_labels2 = getAGE_timeSeries(df, 2)
            sorted_counts3, sorted_labels3 = getAGE_timeSeries(df, 3)
            sorted_counts4, sorted_labels4 = getAGE_timeSeries(df, 4)

            # So far no data for fatalities
            y_dead = []

            # Use 7-day average
            # if use7DayAV:
            #     y_inf_new_7av = np.zeros(y_inf.shape)
            #     for d in range(0, len(y_inf)):
            #         ind = np.arange(d - 3, d + 4)
            #         ind_use = ind[np.logical_and(ind >= 0, ind < len(y_inf))]
            #         y_inf_new_7av[d] = np.mean(np.array(y_new_inf)[ind_use])
            #     y_inf_7av = np.cumsum(y_inf_new_7av)
            #
            #     y_inf = y_inf_7av
            #     y_new_inf = y_inf_new_7av

            # Summarize
            data = []
            data.append(t)
            data.append(y_dead)
            data.append(y_inf)
            data.append(y_new_inf)

        else:

            # read csv files for cases
            os.chdir('/project/data/20200722T103018/content/')  # 20200528T091303/content/epiData')
            filename = 'EpiData_ETH_complete_20200720_extern.xlsx'  # 'EpiData_ETH_censor_20200515.xlsx'
            df = pd.read_excel(filename, 'positive')
            # df_neg = pd.read_excel(filename, 'negative')
            os.chdir(wd)

            # fill in data for Riehen and Bettingen
            df.Quarters.fillna(df.Ortschaft, inplace=True)

            # restrict to dates greater than starting date
            df['ENTNAHMEDATUM'] = pd.to_datetime(df['ENTNAHMEDATUM'], format='%Y-%m-%d')
            df['ENTNAHMEDATUM'] = df['ENTNAHMEDATUM'].dt.date
            START = df['ENTNAHMEDATUM'] >= start
            END = df['ENTNAHMEDATUM'] <= end
            df = df[START & END]
            tmp = (df['ENTNAHMEDATUM'] - start)
            df['DELTA'] = tmp.astype('timedelta64[D]')
            t = np.arange(df['DELTA'].max() + 1)  # np.array(np.unique(df['DELTA']))

            # Get socioeconomic data
            os.chdir(os.path.join(wd, 'geodata'))
            filename = 'SocioeconomicScore_data.csv'
            pop_df = pd.read_csv(filename)
            os.chdir(wd)

            # Get socioeconomic data
            if useNaturalBreaks:
                name_suffix = 'natural_breaks'
            else:
                name_suffix = 'percentiles'
            os.chdir(os.path.join(wd, 'graphs'))
            if useForTiles == 'MedianIncome2017':
                filename = 'bs_MedianIncome2017_' + str(n_splitsSoc) + name_suffix + '.csv'
            elif useForTiles == 'CoHab_index':
                filename = 'bs_CohabProxIndex_' + str(n_splitsSoc) + name_suffix + '.csv'
            elif useForTiles == 'SENIOR_ANT':
                filename = 'bs_SENIOR_ANT_' + str(n_splitsSoc) + name_suffix + '.csv'
            elif useForTiles == 'LivingSpace':
                filename = 'bs_Living_space_per_Person_2017_' + str(n_splitsSoc) + name_suffix + '.csv'
            elif useForTiles == 'random':
                filename = 'bs_random_' + str(n_splitsSoc) + 'tiles_' + randomIndex + '.csv'
            else:
                filename = 'bs_' + useForTiles + '_' + str(n_splitsSoc) + name_suffix + '.csv'

            soc_df = pd.read_csv(filename)
            os.chdir(wd)

            # if the_quarter == 'all':
            #     raise ('The option all is not supported for splitting according to living space')
            # else:
            #
            #     # Get total number of inhabitants
            #     os.chdir(os.path.join(wd, 'geodata'))
            #     filename = 'bs_quarter_mapping_all.csv'
            #     pop_tot_df = pd.read_csv(filename)
            #     os.chdir(wd)
            #
            #     # Split according to living space
            #     pop_df_nan = pop_df.dropna(axis=0)
            #     if the_quarter == 0:
            #         # Use NaNs
            #         blocks = pop_df['BlockID'].loc[pop_df['Living space per Person 2017'].isnull()].values
            #         pop = pop_tot_df['POPULATION'].sum() - pop_df_nan['Population 2017'].sum()
            #     else:
            #         if the_quarter == n_splitsSoc:
            #             cutoff_up = max(pop_df['Living space per Person 2017'])
            #         else:
            #             cutoff_up = np.percentile(pop_df_nan['Living space per Person 2017'],
            #                                       100 / n_splitsSoc * the_quarter)
            #
            #         if the_quarter > 1:
            #             cutoff_low = np.percentile(pop_df_nan['Living space per Person 2017'],
            #                                        100 / n_splitsSoc * (the_quarter - 1))
            #         else:
            #             cutoff_low = 0.9 * np.percentile(pop_df_nan['Living space per Person 2017'], 0)
            #
            #         # Get the blocks belonging to this living space percentile
            #         blocks = pop_df['BlockID'].loc[(cutoff_low < pop_df['Living space per Person 2017']) & (
            #                     pop_df['Living space per Person 2017'] <= cutoff_up)].values
            #
            #         # Get population
            #         pop = pop_df['Population 2017'].loc[pop_df['BlockID'].isin(blocks)].sum()
            #
            #     try:
            #         # Subset the case data
            #         df = df[df['BlockID'].isin(list(blocks))]
            #         print(the_quarter + " has a total of " + str(df.shape[0]) + ' cases')
            #     except:
            #         raise('Subset for quarter failed!!!')

            # Produce time series
            if the_quarter == 'all':
                raise ('The option all is not supported for splitting according to living space')
            else:

                # Get total number of inhabitants
                os.chdir(os.path.join(wd, 'geodata'))
                filename = 'bs_quarter_mapping_all.csv'
                pop_tot_df = pd.read_csv(filename)
                os.chdir(wd)

                # For each tile get the blocks and population
                blocks = soc_df['BLO_ID'].loc[soc_df['percentile'] == the_quarter].values

                # Split according to living space
                pop_df_nan = pop_df.dropna(axis=0)
                if the_quarter == 0:

                    # # Use NaNs
                    # blocks = pop_df['BlockID'].loc[pop_df['Living space per Person 2017'].isnull()].values
                    # pop = pop_tot_df['POPULATION'].sum() - pop_df_nan['Population 2017'].sum()

                    otherBlocks = soc_df['BLO_ID'].loc[soc_df['percentile'] != the_quarter].values
                    pop = pop_tot_df['POPULATION'].sum() - pop_df['Population 2017'].loc[
                        pop_df['BlockID'].isin(otherBlocks)].sum()
                else:
                    # if the_quarter == n_splitsSoc:
                    #     cutoff_up = max(pop_df['Living space per Person 2017'])
                    # else:
                    #     cutoff_up = np.percentile(pop_df_nan['Living space per Person 2017'], 100 / n_splitsSoc * the_quarter)
                    #
                    # if the_quarter>1:
                    #     cutoff_low = np.percentile(pop_df_nan['Living space per Person 2017'], 100 / n_splitsSoc * (the_quarter-1))
                    # else:
                    #     cutoff_low = 0.9*np.percentile(pop_df_nan['Living space per Person 2017'],0)
                    #
                    #
                    # # Get the blocks belonging to this living space percentile
                    # blocks = pop_df['BlockID'].loc[(cutoff_low < pop_df['Living space per Person 2017']) & (pop_df['Living space per Person 2017']<=cutoff_up)].values

                    # Get population
                    pop = pop_df['Population 2017'].loc[pop_df['BlockID'].isin(blocks)].sum()

                try:
                    # Subset the case data
                    df = df[df['BlockID'].isin(list(blocks))]
                    print(the_quarter + " has a total of " + str(df.shape[0]) + ' cases')
                except:
                    raise ('Subset for quarter failed!!!')

            counts_del = df['DELTA'].value_counts().values
            labels_del = df['DELTA'].value_counts().keys().values.astype('timedelta64[D]')
            y_new_inf = [x for _, x in sorted(zip(labels_del, counts_del))]
            tsub = np.sort(labels_del)
            tsub = tsub.astype('timedelta64[D]') / np.timedelta64(1, 'D')
            for k in range(0, len(t)):
                if t[k] not in tsub:
                    y_new_inf.insert(k, 0)

            y_inf = np.cumsum(y_new_inf)

            # Subset age group (not used at this stage yet)
            sorted_counts1, sorted_labels1 = getAGE_timeSeries(df, 1)
            sorted_counts2, sorted_labels2 = getAGE_timeSeries(df, 2)
            sorted_counts3, sorted_labels3 = getAGE_timeSeries(df, 3)
            sorted_counts4, sorted_labels4 = getAGE_timeSeries(df, 4)

            # So far no data for fatalities
            y_dead = []

            # Summarize
            data = []
            data.append(t)
            data.append(y_dead)
            data.append(y_inf)
            data.append(y_new_inf)

    else:
        if usePangolin:
            # read csv files for cases
            os.chdir('/project/data/20200817T141323/content/')  # 20200528T091303/content/epiData')
            filename = 'EpiData_ETH_complete_20200816_jElHFB.xlsx'  # 'EpiData_ETH_censor_20200515.xlsx'
            df = pd.read_excel(filename, 'positive')
            # df_neg = pd.read_excel(filename, 'negative')
            os.chdir(wd)

            # Subset for the pangolin strain
            df = df[df['Pangolin'] == pangolingStrain]

            # fill in data for Riehen and Bettingen
            df.Quarters.fillna(df.Ortschaft, inplace=True)

            # read soc'ioeconomic data
            os.chdir(os.path.join(wd, 'geodata'))
            # filename = 'SocioeconomicScore_data.xls'
            filename = 'bs_quarter_mapping_all.csv'
            pop_df = pd.read_csv(filename)
            os.chdir(wd)

            # restrict to dates greater than starting date
            df['ENTNAHMEDATUM'] = pd.to_datetime(df['ENTNAHMEDATUM'], format='%Y-%m-%d')
            df['ENTNAHMEDATUM'] = df['ENTNAHMEDATUM'].dt.date
            START = df['ENTNAHMEDATUM'] >= start
            END = df['ENTNAHMEDATUM'] <= end
            df = df[START & END]
            tmp = (df['ENTNAHMEDATUM'] - start)
            df['DELTA'] = tmp.astype('timedelta64[D]')
            t = np.arange(df['DELTA'].max() + 1)  # np.array(np.unique(df['DELTA']))

            # Subset for selected adm1_pcode
            if the_quarter == 'all':
                pop = pop_df['POPULATION'].sum()
            else:

                # Get the quarters beloning to this ID
                quarters = pop_df['GEBIET'].loc[pop_df['ID_PARTNER'] == the_quarter].values
                print('Using these quarters:', quarters)

                # Get population
                pop = pop_df['POPULATION'].loc[pop_df['ID_PARTNER'] == the_quarter].sum()

                try:
                    # Subset the case data
                    df = df[df['Quarters'].isin(list(quarters))]
                    print(quarters + " has a total of " + str(df.shape[0]) + ' cases')
                except:
                    raise ('Subset for quarter failed!!!')

            # Produce time series
            counts_del = df['DELTA'].value_counts().values
            labels_del = df['DELTA'].value_counts().keys().values.astype('timedelta64[D]')
            y_new_inf = [x for _, x in sorted(zip(labels_del, counts_del))]
            tsub = np.sort(labels_del)
            tsub = tsub.astype('timedelta64[D]') / np.timedelta64(1, 'D')
            for k in range(0, len(t)):
                if t[k] not in tsub:
                    y_new_inf.insert(k, 0)

            y_inf = np.cumsum(y_new_inf)

            # Subset age group (not used at this stage yet)
            sorted_counts1, sorted_labels1 = getAGE_timeSeries(df, 1)
            sorted_counts2, sorted_labels2 = getAGE_timeSeries(df, 2)
            sorted_counts3, sorted_labels3 = getAGE_timeSeries(df, 3)
            sorted_counts4, sorted_labels4 = getAGE_timeSeries(df, 4)

            # So far no data for fatalities
            y_dead = []

            # Summarize
            data = []
            data.append(t)
            data.append(y_dead)
            data.append(y_inf)
            data.append(y_new_inf)

        else:
            # read csv files for cases
            os.chdir('/project/data/20200722T103018/content/')  # 20200528T091303/content/epiData')
            filename = 'EpiData_ETH_complete_20200720_extern.xlsx'  # 'EpiData_ETH_censor_20200515.xlsx'
            df = pd.read_excel(filename, 'positive')
            # df_neg = pd.read_excel(filename, 'negative')
            os.chdir(wd)

            # fill in data for Riehen and Bettingen
            df.Quarters.fillna(df.Ortschaft, inplace=True)

            # read soc'ioeconomic data
            os.chdir(os.path.join(wd, 'geodata'))
            # filename = 'SocioeconomicScore_data.xls'
            filename = 'bs_quarter_mapping_all.csv'
            pop_df = pd.read_csv(filename)
            os.chdir(wd)

            # restrict to dates greater than starting date
            df['ENTNAHMEDATUM'] = pd.to_datetime(df['ENTNAHMEDATUM'], format='%Y-%m-%d')
            df['ENTNAHMEDATUM'] = df['ENTNAHMEDATUM'].dt.date
            START = df['ENTNAHMEDATUM'] >= start
            END = df['ENTNAHMEDATUM'] <= end
            df = df[START & END]
            tmp = (df['ENTNAHMEDATUM'] - start)
            df['DELTA'] = tmp.astype('timedelta64[D]')
            t = np.arange(df['DELTA'].max() + 1)  # np.array(np.unique(df['DELTA']))

            # Subset for selected adm1_pcode
            if the_quarter == 'all':
                pop = pop_df['POPULATION'].sum()
            else:

                # Get the quarters beloning to this ID
                quarters = pop_df['GEBIET'].loc[pop_df['ID_PARTNER'] == the_quarter].values
                print('Using these quarters:', quarters)

                # Get population
                pop = pop_df['POPULATION'].loc[pop_df['ID_PARTNER'] == the_quarter].sum()

                try:
                    # Subset the case data
                    df = df[df['Quarters'].isin(list(quarters))]
                    print(quarters + " has a total of " + str(df.shape[0]) + ' cases')
                except:
                    print('Using all data since subset for quarter failed!!!')

            # Produce time series
            counts_del = df['DELTA'].value_counts().values
            labels_del = df['DELTA'].value_counts().keys().values.astype('timedelta64[D]')
            y_new_inf = [x for _, x in sorted(zip(labels_del, counts_del))]
            tsub = np.sort(labels_del)
            tsub = tsub.astype('timedelta64[D]') / np.timedelta64(1, 'D')
            for k in range(0, len(t)):
                if t[k] not in tsub:
                    y_new_inf.insert(k, 0)

            y_inf = np.cumsum(y_new_inf)

            # Subset age group (not used at this stage yet)
            sorted_counts1, sorted_labels1 = getAGE_timeSeries(df, 1)
            sorted_counts2, sorted_labels2 = getAGE_timeSeries(df, 2)
            sorted_counts3, sorted_labels3 = getAGE_timeSeries(df, 3)
            sorted_counts4, sorted_labels4 = getAGE_timeSeries(df, 4)

            # So far no data for fatalities
            y_dead = []

            # Summarize
            data = []
            data.append(t)
            data.append(y_dead)
            data.append(y_inf)
            data.append(y_new_inf)

    return data, pop, baselstrain, sequenced, posCases, t_poscases


def obtain_adjacencyILGE(quarter, neigh, n_splitsSoc):
    global useForTiles

    if neigh == 'physical':
        file = os.path.join(wd, 'graphs', 'bs_physical_neighborhood.csv')
        A_all = pd.read_csv(file)

        if quarter[0] == 'all':
            A = A_all
        else:
            A = A_all[[str(i) for i in ['Unnamed: 0'] + quarter]].sort_values(by=['Unnamed: 0'])
            A = A.set_index('Unnamed: 0')
            A = A.loc[quarter, :]
    elif neigh == 'estimate':
        A = np.ones((len(quarter), len(quarter)))

    elif neigh == 'transport':
        file = os.path.join(wd, 'graphs', 'bs_sbb_quarter.csv')
        A_all = pd.read_csv(file)

        if quarter[0] == 'all':
            A = A_all
        else:
            A = A_all[[str(i) for i in ['Unnamed: 0'] + quarter]].sort_values(by=['Unnamed: 0'])
            A = A.set_index('Unnamed: 0')
            A = A.loc[quarter, :]

        # Normalize
        row_sums = A.sum(axis=1)
        A = A / A.sum().sum()

        # Remove diagonal elements
        A = np.fill_diagonal(A, 0)

    elif neigh == 'merged':

        transport_means = ['publ', 'bike', 'moto', 'foot']
        if useNaturalBreaks:
            name_suffix = 'natural_breaks'
        else:
            name_suffix = 'percentiles'

        for i, tr in enumerate(transport_means):

            if useForTiles == 'MedianIncome2017':
                file = os.path.join(wd, 'graphs', 'bs_MedianIncome2017_' + str(
                    n_splitsSoc) + name_suffix + '_' + tr + '_mobility.csv')
            elif useForTiles == 'CoHab_index':
                file = os.path.join(wd, 'graphs', 'bs_CohabProxIndex_' + str(
                    n_splitsSoc) + name_suffix + '_' + tr + '_mobility.csv')
            elif useForTiles == 'SENIOR_ANT':
                file = os.path.join(wd, 'graphs', 'bs_SENIOR_ANT_' + str(
                    n_splitsSoc) + name_suffix + '_' + tr + '_mobility.csv')
            elif useForTiles == 'LivingSpace':
                file = os.path.join(wd, 'graphs', 'bs_Living_space_per_Person_2017_' + str(
                    n_splitsSoc) + name_suffix + '_' + tr + '_mobility.csv')
            elif useForTiles == 'random':
                file = os.path.join(wd, 'graphs', 'bs_random_' + str(
                    n_splitsSoc) + 'tiles_' + randomIndex + '_' + tr + '_mobility.csv')
            else:
                file = os.path.join(wd, 'graphs', 'bs_' + useForTiles + '_' + str(
                    n_splitsSoc) + name_suffix + '_' + tr + '_mobility.csv')

            A_all = pd.read_csv(file)

            if quarter[0] == 'all':
                A_tr = A_all
            else:
                A_tr = A_all[[str(i) for i in ['Unnamed: 0'] + quarter]].sort_values(by=['Unnamed: 0'])
                A_tr = A_tr.set_index('Unnamed: 0')
                A_tr = A_tr.loc[quarter, :]

            # Sum up
            if i == 0:
                A = A_tr
            else:
                A = A + A_tr
                # row_sums = A.sum(axis=1)
                # A = A / row_sums[:, np.newaxis]

        # Remove diagonal elements
        if diagonalMobilityZero:
            np.fill_diagonal(A_tr.values, 0)

        # Normalize
        A = A_tr / A_tr.sum().sum()



    else:
        raise ('Invalid neighbor!')

    return A


def obtain_adjacencyILGE_NormRow(quarter, neigh, n_splitsSoc, pop):
    global useForTiles

    if neigh == 'physical':
        file = os.path.join(wd, 'graphs', 'bs_physical_neighborhood.csv')
        A_all = pd.read_csv(file)

        if quarter[0] == 'all':
            A = A_all
        else:
            A = A_all[[str(i) for i in ['Unnamed: 0'] + quarter]].sort_values(by=['Unnamed: 0'])
            A = A.set_index('Unnamed: 0')
            A = A.loc[quarter, :]
    elif neigh == 'estimate':
        A = np.ones((len(quarter), len(quarter)))

    elif neigh == 'transport':
        file = os.path.join(wd, 'graphs', 'bs_sbb_quarter.csv')
        A_all = pd.read_csv(file)

        if quarter[0] == 'all':
            A = A_all
        else:
            A = A_all[[str(i) for i in ['Unnamed: 0'] + quarter]].sort_values(by=['Unnamed: 0'])
            A = A.set_index('Unnamed: 0')
            A = A.loc[quarter, :]

        # Normalize
        row_sums = A.sum(axis=1)
        A = A / A.sum().sum()

        # Remove diagonal elements
        A = np.fill_diagonal(A, 0)

    elif neigh == 'merged':

        transport_means = ['publ', 'bike', 'moto', 'foot']
        if useNaturalBreaks:
            name_suffix = 'natural_breaks'
        else:
            name_suffix = 'percentiles'

        for i, tr in enumerate(transport_means):

            if useForTiles == 'MedianIncome2017':
                file = os.path.join(wd, 'graphs', 'bs_MedianIncome2017_' + str(
                    n_splitsSoc) + name_suffix + '_' + tr + '_mobility.csv')
            elif useForTiles == 'CoHab_index':
                file = os.path.join(wd, 'graphs', 'bs_CohabProxIndex_' + str(
                    n_splitsSoc) + name_suffix + '_' + tr + '_mobility.csv')
            elif useForTiles == 'SENIOR_ANT':
                file = os.path.join(wd, 'graphs', 'bs_SENIOR_ANT_' + str(
                    n_splitsSoc) + name_suffix + '_' + tr + '_mobility.csv')
            elif useForTiles == 'LivingSpace':
                file = os.path.join(wd, 'graphs', 'bs_Living_space_per_Person_2017_' + str(
                    n_splitsSoc) + name_suffix + '_' + tr + '_mobility.csv')
            elif useForTiles == 'random':
                file = os.path.join(wd, 'graphs', 'bs_random_' + str(
                    n_splitsSoc) + 'tiles_' + randomIndex + '_' + tr + '_mobility.csv')
            else:
                file = os.path.join(wd, 'graphs', 'bs_' + useForTiles + '_' + str(
                    n_splitsSoc) + name_suffix + '_' + tr + '_mobility.csv')

            A_all = pd.read_csv(file)

            if quarter[0] == 'all':
                A_tr = A_all
            else:
                A_tr = A_all[[str(i) for i in ['Unnamed: 0'] + quarter]].sort_values(by=['Unnamed: 0'])
                A_tr = A_tr.set_index('Unnamed: 0')
                A_tr = A_tr.loc[quarter, :]

            # Sum up
            if i == 0:
                A = A_tr
            else:
                A = A + A_tr

        # Remove diagonal elements
        if diagonalMobilityZero:
            np.fill_diagonal(A_tr.values, 0)

        # Normalize to 1 in total
        Anorm = np.array(A_tr / A_tr.sum().sum())

        # Normalize each row
        pop_frac = sum(pop) / pop
        n_adm = len(pop)
        A = np.zeros((Anorm.shape))
        for k in range(0, n_adm):
            for j in range(0, n_adm):
                A[k, j] = (Anorm[k, j] + Anorm[j, k]) * pop[j] / (pop[j] + pop[k])


    else:
        raise ('Invalid neighbor!')

    return A


def obtain_SocAdjacencyILGE(quarter):
    file = os.path.join(wd, 'graphs', 'bs_socioeco_median-distance.csv')
    A_all = pd.read_csv(file)

    if quarter[0] == 'all':
        A = A_all
    else:
        A = A_all[[str(i) for i in ['Unnamed: 0'] + quarter]].sort_values(by=['Unnamed: 0'])
        A = A.set_index('Unnamed: 0')
        A = A.loc[quarter, :]

    # Normalize
    row_sums = A.sum(axis=1)
    A = A / A.sum().sum()

    return A


def getProximityIndex():
    n_tiles = 3

    # get data
    file = 'ProximityIndex_200929.xls'
    path = os.path.join(wd, 'geodata', file)
    df = pd.read_excel(path)

    # Counts per quantile
    cp_index = df['CohabProxInd'].values
    # plt.hist(cp_index, 20)
    # plt.xlabel('CP index')
    # plt.ylabel('Number of Blocks')

    # Divide
    cp_3tiles = np.zeros(cp_index.shape)
    for i in range(0, len(df['BlockID'])):

        if cp_index[i] <= 4:
            cp_3tiles[i] = 1
        elif 4 < cp_index[i] <= 6:
            cp_3tiles[i] = 2
        elif 6 < cp_index[i]:
            cp_3tiles[i] = 3
    plt.hist(cp_3tiles, 10)
    plt.xlabel('CP Tile')
    plt.ylabel('Number of Blocks')
    plt.title('3 Tiles')

    cp_4tiles = np.zeros(cp_index.shape)
    for i in range(0, len(df['BlockID'])):

        if cp_index[i] <= 4:
            cp_4tiles[i] = 1
        elif 4 < cp_index[i] <= 5:
            cp_4tiles[i] = 2
        elif 5 < cp_index[i] <= 6:
            cp_4tiles[i] = 3
        elif 6 < cp_index[i]:
            cp_4tiles[i] = 4
    plt.hist(cp_4tiles, 10)
    plt.xlabel('CP Tile')
    plt.ylabel('Number of Blocks')
    plt.title('4 Tiles')

    cp_5tiles = np.zeros(cp_index.shape)
    for i in range(0, len(df['BlockID'])):

        if cp_index[i] <= 3:
            cp_5tiles[i] = 1
        elif 3 < cp_index[i] <= 4:
            cp_5tiles[i] = 2
        elif 4 < cp_index[i] <= 5:
            cp_5tiles[i] = 3
        elif 5 < cp_index[i] <= 6:
            cp_5tiles[i] = 4
        elif 6 < cp_index[i]:
            cp_5tiles[i] = 5
    plt.hist(cp_5tiles, 10)
    plt.xlabel('CP Tile')
    plt.ylabel('Number of Blocks')
    plt.title('5 Tiles')

    # save
    df = df.rename(columns={"BlockID": "BLO_ID"})
    df3tiles = df.copy()
    df3tiles['percentile'] = cp_3tiles
    df3tiles.to_csv(os.path.join(wd, 'graphs', 'cohabitationProxIndex_3Tiles.csv'))
    df4tiles = df.copy()
    df4tiles['percentile'] = cp_4tiles
    df4tiles.to_csv(os.path.join(wd, 'graphs', 'cohabitationProxIndex_4Tiles.csv'))
    df5tiles = df.copy()
    df5tiles['percentile'] = cp_5tiles
    df5tiles.to_csv(os.path.join(wd, 'graphs', 'cohabitationProxIndex_5Tiles.csv'))

    np.nanquantile(cp_index, 0.33)


def getProximityIndex_Jenks():
    n_tiles = 3

    # get data
    file = 'ProximityIndex_200929.xls'
    path = os.path.join(wd, 'geodata', file)
    df = pd.read_excel(path)

    # get data
    lSp = df['LivSpacePerPers17'].values
    PpHH = df['Anteil 2- und Mehr-PH'].values

    # Exclude outlier
    ind = np.where(lSp > 311)[0][0]
    lSp[ind] = np.nan

    # 1. Exclude Nans
    nonaninds = np.logical_and(~np.isnan(lSp), ~np.isnan(PpHH))
    lSp_noNan = lSp[nonaninds]
    PpHH_noNan = PpHH[nonaninds]

    # Normalized living space data
    n_lSp_noNan = (lSp_noNan - lSp_noNan.min()) / (np.nanmax(lSp_noNan - np.nanmin(lSp_noNan)))
    n_lSp = (lSp - lSp_noNan.min()) / (np.nanmax(lSp - np.nanmin(lSp_noNan)))

    # 2+PpHH - already normalized
    n_2PpHH_noNan = PpHH_noNan

    # Check correlation of Living space and 2+PpHH:
    rho, p = stats.spearmanr(n_lSp_noNan, n_2PpHH_noNan)

    # Plot
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(n_lSp_noNan, n_2PpHH_noNan)
    plt.xlabel('normalized Living space pP')
    plt.ylabel('percentage of 2+ person house holds')

    # Sum living space and +2personHH
    sum_lsp_2pHH = PpHH + n_lSp

    # Back to dataframe
    df['Sum_Lsp_2pHH'] = sum_lsp_2pHH

    # Calcualte natural breaks according to tile numbers
    breaks = jenkspy.jenks_breaks(df['Sum_Lsp_2pHH'], nb_class=n_tiles)
    breaks4 = jenkspy.jenks_breaks(df['Sum_Lsp_2pHH'], nb_class=4)
    breaks5 = jenkspy.jenks_breaks(df['Sum_Lsp_2pHH'], nb_class=5)

    # Divide
    cp_index = df['Sum_Lsp_2pHH'].values
    cp_3tiles = np.zeros(lSp.shape)
    for i in range(0, len(df['BlockID'])):

        if cp_index[i] <= breaks[1]:
            cp_3tiles[i] = 1
        elif breaks[1] < cp_index[i] <= breaks[2]:
            cp_3tiles[i] = 2
        elif breaks[2] < cp_index[i]:
            cp_3tiles[i] = 3

    fig1 = plt.figure(figsize=(6, 6))
    plt.hist(cp_3tiles, 10)
    plt.xlabel('CP Tile')
    plt.ylabel('Number of Blocks')
    plt.title('3 Tiles')

    cp_4tiles = np.zeros(cp_index.shape)
    for i in range(0, len(df['BlockID'])):

        if cp_index[i] <= breaks4[1]:
            cp_4tiles[i] = 1
        elif breaks4[1] < cp_index[i] <= breaks4[2]:
            cp_4tiles[i] = 2
        elif breaks4[2] < cp_index[i] <= breaks4[3]:
            cp_4tiles[i] = 3
        elif breaks4[3] < cp_index[i]:
            cp_4tiles[i] = 4
    fig1 = plt.figure(figsize=(6, 6))
    plt.hist(cp_4tiles, 10)
    plt.xlabel('CP Tile')
    plt.ylabel('Number of Blocks')
    plt.title('4 Tiles')

    cp_5tiles = np.zeros(cp_index.shape)
    for i in range(0, len(df['BlockID'])):

        if cp_index[i] <= breaks5[1]:
            cp_5tiles[i] = 1
        elif breaks5[1] < cp_index[i] <= breaks5[2]:
            cp_5tiles[i] = 2
        elif breaks5[2] < cp_index[i] <= breaks5[3]:
            cp_5tiles[i] = 3
        elif breaks5[3] < cp_index[i] <= breaks5[4]:
            cp_5tiles[i] = 4
        elif breaks5[4] < cp_index[i]:
            cp_5tiles[i] = 5
    fig1 = plt.figure(figsize=(6, 6))
    plt.hist(cp_5tiles, 10)
    plt.xlabel('CP Tile')
    plt.ylabel('Number of Blocks')
    plt.title('5 Tiles')

    # save
    df = df.rename(columns={"BlockID": "BLO_ID"})
    df3tiles = df.copy()
    df3tiles['percentile'] = cp_3tiles
    df3tiles.to_csv(os.path.join(wd, 'graphs', 'cohabitationProxIndex_3Tiles.csv'))
    df4tiles = df.copy()
    df4tiles['percentile'] = cp_4tiles
    df4tiles.to_csv(os.path.join(wd, 'graphs', 'cohabitationProxIndex_4Tiles.csv'))
    df5tiles = df.copy()
    df5tiles['percentile'] = cp_5tiles
    df5tiles.to_csv(os.path.join(wd, 'graphs', 'cohabitationProxIndex_5Tiles.csv'))

    np.nanquantile(cp_index, 0.33)


# Parameters and bounds
def par_list_general(model, n_adm, n_exp_0, n_inf_0, n_rec_0, n_fat_0, n_asi_0, \
                     n_asr_0, n_und_0, R_infU_in_0, T_infI_0, T_inc_0, p_fat_0, R_red_0, R_redU_0, p_sym_0, \
                     R_asy_in_0, b_deR_0, a_deR_0, T_infA_0, T_infU_0, alpha_0, alpha_f, a_alpha, b_alpha, adj_el,
                     local, neigh, alpha_fix, alphaS_0, R_i_asy, T_i_asy, p_asy, n_un_i_0, n_un_r_0, seedSingle,
                     ind_seedQuarter):
    if model == 'SEAIRD':
        par_list = initial_vars(n_adm, n_exp_0, n_inf_0, n_rec_0, n_fat_0, n_asi_0, \
                                n_asr_0, n_und_0)
        par_list += [R_infU_in_0, T_infI_0, T_inc_0, p_fat_0, R_red_0, R_redU_0, p_sym_0, \
                     R_asy_in_0, b_deR_0, a_deR_0, T_infA_0, T_infU_0]
    elif model == 'SEUI':

        if alpha_fix:

            # Mobility rate alpha is fixed but free, R is time-dependent
            print('R varies over time, alpha constant over time')
            par_list = initial_vars_SEUI(n_adm, n_exp_0, n_und_0, n_inf_0, n_un_i_0, n_un_r_0, seedSingle,
                                         ind_seedQuarter)
            if local:
                print('All parameters individually optimized for each quarter!')

                if useRelativeR:
                    par_list += [R_infU_in_0] + list([0.99, 0.46]) + [T_inc_0] + list(np.repeat(R_redU_0, n_adm)) + \
                                list(np.repeat(b_deR_0, n_adm)) + list(np.repeat(a_deR_0, n_adm)) \
                                + [T_infU_0, alpha_0, alphaS_0] \
                                + list(np.repeat(R_i_asy, n_adm)) + [T_i_asy, p_asy]

                else:
                    par_list += list(np.repeat(R_infU_in_0, n_adm)) + [T_inc_0] + list(np.repeat(R_redU_0, n_adm)) + \
                                list(np.repeat(b_deR_0, n_adm)) + list(np.repeat(a_deR_0, n_adm)) \
                                + [T_infU_0, alpha_0, alphaS_0] \
                                + list(np.repeat(R_i_asy, n_adm)) + [T_i_asy, p_asy]
            else:
                par_list += list(np.repeat(R_infU_in_0, n_adm)) + [T_inc_0] + list(np.repeat(R_redU_0, n_adm)) + \
                            list(np.repeat(b_deR_0, n_adm)) + list(np.repeat(a_deR_0, n_adm)) \
                            + [T_infU_0, alpha_0, alphaS_0] \
                            + list(np.repeat(R_i_asy, n_adm)) + [T_i_asy, p_asy]
        else:

            # Mobility rate alpha is time dependent, but R not
            print('Alpha varies over time, R constant over time')
            par_list = initial_vars_SEUI(n_adm, n_exp_0, n_und_0, n_inf_0)
            if local:
                print('All parameters individually optimized for each quarter!')
                # separate set of parameters for all
                par_list += list(
                    np.repeat([R_infU_in_0, T_inc_0, alpha_f, b_alpha, a_alpha, T_infU_0, alpha_0, alphaS_0], n_adm))
            else:
                print('All parameters the same but optimized for all quarters!')
                if neigh == 'estimate':
                    print('Fitting adjacency matrix elements!')
                    par_list += [R_infU_in_0, T_inc_0, alpha_f, b_alpha, a_alpha, T_infU_0, alpha_0] + \
                                list(adj_el * np.ones((n_adm * n_adm, 1)))
                else:
                    print('NOT fitting adjacency matrix elements!')
                    par_list += list(np.repeat(R_infU_in_0, n_adm)) + [T_inc_0] + list(np.repeat(alpha_f, n_adm)) + \
                                list(np.repeat(b_alpha, n_adm)) + list(np.repeat(a_alpha, n_adm)) \
                                + [T_infU_0, alpha_0, alphaS_0]


    else:
        raise ('Invalid ODE model')

    return par_list


def bnds_vars_general(n_adm, model, bnd_n_exp, bnd_n_inf, bnd_n_rec, bnd_n_fat, bnd_n_asi, bnd_n_asr, \
                      bnd_n_und, bnd_R_infU_in, bnd_T_infI, bnd_T_inc, bnd_p_fat, bnd_R_red, bnd_R_redU, \
                      bnd_p_sym, bnd_R_asy_in, bnd_b_deR, bnd_a_deR, bnd_T_infA, bnd_T_infU, bnd_alpha,
                      bnd_alpha_f, bnd_a_alpha, bnd_b_alpha, bnd_adj_el, local, neigh, alpha_fix, bnd_alphaS,
                      bnd_n_uni, bnd_n_unr, bnd_R_i_asy, bnd_T_i_asy, bnd_p_asy, seedSingle, seedQuarter):
    bnds_lst = [bnd_n_exp, bnd_n_inf, bnd_n_rec, bnd_n_fat, bnd_n_asi, bnd_n_asr, \
                bnd_n_und]

    if model == 'SEAIRD':
        bnds = bnds_vars(n_adm, bnds_lst)
        bnds += bnd_R_infU_in + bnd_T_infI + bnd_T_inc + bnd_p_fat + bnd_R_red + bnd_R_redU + \
                bnd_p_sym + bnd_R_asy_in + bnd_b_deR + bnd_a_deR + bnd_T_infA + bnd_T_infU


    elif model == 'SEUI':
        bnds = bnds_vars_SEUI(n_adm, bnds_lst, bnd_n_uni, bnd_n_unr, seedSingle, seedQuarter)

        if alpha_fix:

            # Mobility rate alpha is fixed but free, R is time-dependent
            if local:

                if useRelativeR:

                    bnds += bnd_R_infU_in + ((0., 1.7),) * (
                                n_adm - 1) + bnd_T_inc + bnd_R_redU * n_adm + bnd_b_deR * n_adm + \
                            bnd_a_deR * n_adm + bnd_T_infU + bnd_alpha + bnd_alphaS + \
                            bnd_R_i_asy * n_adm + bnd_T_i_asy + bnd_p_asy

                else:
                    bnds += bnd_R_infU_in * n_adm + bnd_T_inc + bnd_R_redU * n_adm + bnd_b_deR * n_adm + \
                            bnd_a_deR * n_adm + bnd_T_infU + bnd_alpha + bnd_alphaS + \
                            bnd_R_i_asy * n_adm + bnd_T_i_asy + bnd_p_asy
            else:
                bnds += bnd_R_infU_in * n_adm + bnd_T_inc + bnd_R_redU * n_adm + bnd_b_deR * n_adm + \
                        bnd_a_deR * n_adm + bnd_T_infU + bnd_alpha + bnd_alphaS + \
                        bnd_R_i_asy * n_adm + bnd_T_i_asy + bnd_p_asy
        else:

            # Mobility rate alpha is time dependent, but R not
            if local:
                # All parameters individually optimized for each quarter
                # separate set of parameters for all
                bnds += bnd_R_infU_in * n_adm + bnd_T_inc + bnd_alpha_f * n_adm + bnd_b_alpha * n_adm + \
                        bnd_a_alpha * n_adm + bnd_T_infU + bnd_alpha * n_adm
            else:
                # All parameters the same but optimized for all quarters
                if neigh == 'estimate':
                    # Fitting adjacency matrix elements
                    bnds += bnd_R_infU_in + bnd_T_inc + bnd_alpha_f + bnd_b_alpha + bnd_a_alpha + bnd_T_infU + \
                            bnd_alpha + bnd_adj_el * n_adm * n_adm
                else:
                    # NOT fitting adjacency matrix elements
                    bnds += bnd_R_infU_in * n_adm + bnd_T_inc + bnd_alpha_f * n_adm + bnd_b_alpha * n_adm + \
                            bnd_a_alpha * n_adm + bnd_T_infU + bnd_alpha


    else:
        raise ('Invalid ODE model')

    return bnds


def initial_vars_SEUI(n_adm, n_exp_0, n_und_0, n_inf_0, n_un_i_0, n_un_r_0, seedSingle, ind_seedQuarter):
    '''Defines initial condition for each population compartment in each area.

    Parameters
    ----------
    n_adm   : number of areas/nodes considered
    n_xxx_0 : initial condition for compartments

    Returns
    -------
    list containing all the initial conditions.
    '''
    if seedSingle:

        # No cases in any quarter but one
        n_exp_0 = 0
        n_und_0 = 0
        n_inf_0 = 0
        n_un_i_0 = 0
        n_un_r_0 = 0
        n_r_0 = 0

        if useReortingDelay:
            inits = [n_exp_0, n_und_0, n_r_0, n_un_i_0, n_un_r_0, n_inf_0]
            inits_all_adm = np.repeat(inits, n_adm)
            inits_all_adm = inits_all_adm.tolist()
            inits_all_adm[ind_seedQuarter] = 1.
        else:
            inits = [n_exp_0, n_und_0, n_inf_0, n_un_i_0, n_un_r_0]
            inits_all_adm = np.repeat(inits, n_adm)
            inits_all_adm = inits_all_adm.tolist()
            inits_all_adm[ind_seedQuarter] = 5.

    else:
        if useReortingDelay:
            inits = [n_exp_0, n_und_0, n_r_0, n_un_i_0, n_un_r_0, n_inf_0]
            inits_all_adm = np.repeat(inits, n_adm)
            inits_all_adm = inits_all_adm.tolist()
        else:
            inits = [n_exp_0, n_und_0, n_inf_0, n_un_i_0, n_un_r_0]
            inits_all_adm = np.repeat(inits, n_adm)
            inits_all_adm = inits_all_adm.tolist()
    return inits_all_adm


def bnds_vars_SEUI(n_adm, bnds_lst, bnd_n_uni, bnd_n_unr, seedSingle, seedQuarter):
    '''Defines the bounds for each compartment variable. Repeats the same
    bounds for each variable to be the same for all the considered adm areas.

    Parameters
    ----------
    n_adm     : number of nodes/areas.
    bnd_n_xxx : single bounds for initial compartment size

    Returns
    -------
    list containing the bounds for the state variables.
    '''
    [bnd_n_exp, bnd_n_inf, bnd_n_rec, bnd_n_fat, bnd_n_asi, bnd_n_asr, bnd_n_in2] \
        = bnds_lst

    if seedSingle:
        bnd_n_uni = ((0, 0),)
        bnd_n_unr = ((0, 0),)
        bnd_n_exp = ((0, 0),)
        bnd_n_inf = ((0, 0),)
        bnd_n_in2 = ((0, 0),)
        bnd_n_r = ((0, 0),)
        if useReortingDelay:
            bnds = bnd_n_exp * n_adm + bnd_n_r * n_adm + bnd_n_inf * n_adm + bnd_n_uni * n_adm + bnd_n_unr * n_adm + bnd_n_in2 * n_adm
        else:
            bnds = bnd_n_exp * n_adm + bnd_n_in2 * n_adm + bnd_n_inf * n_adm + bnd_n_uni * n_adm + bnd_n_unr * n_adm
        lst_bnds = list(bnds)
        lst_bnds[seedQuarter] = (0.1, 5)
        bnds = tuple(lst_bnds)
    else:
        bnds = bnd_n_exp * n_adm + bnd_n_in2 * n_adm + bnd_n_inf * n_adm + bnd_n_uni * n_adm + bnd_n_unr * n_adm

    return bnds


# Fit
def fit_general(model, par_list, data_trn, bnds, fixed_pars, adm0, t_trn, t_tst, alpha_fix, neigh):
    if model == 'SEAIRD':
        result = dofit2(par_list, data_trn, bnds, fixed_pars, adm0)

        # optain curve resulting from optimization
        t = np.concatenate((t_trn, t_tst))
        fit = solution2(t, result.x, fixed_pars).T

    elif model == 'SEUI':

        result = dofit_SEUI(par_list, data_trn, bnds, fixed_pars, adm0)

        # optain curve resulting from optimization
        t = np.concatenate((t_trn, t_tst))
        fit = solution_SEUI(t, result.x, fixed_pars).T

    else:
        raise ('Invalid ODE model')

    return result, t, fit


def ode_model_SEUI(Y, t, pars, fixed_pars):
    """ODE model. The order of compartments in Y is s, e, i, r, d, a, ar, i2,
    repeated for the number of considered nodes. If 3 adm1 areas are cosidered,
    than y is: s1, s2, s3, e1, e2, ... and so on.

    Parameters
    ----------
    Y          : vector containing the values of the state variables:
    t          : time vector
    pars       : vector containing the model parameters' current values
    fixed_pars : vector containing the fixed parameters

    Returns
    -------
    list of values of the differentials for each compartment variable at time t
    """

    n_adm = fixed_pars[0]
    c = fixed_pars[4]
    local = fixed_pars[5]
    neigh = fixed_pars[7]
    alp_fix = fixed_pars[8]
    loadFr_dyn = fixed_pars[10]
    time_dep = fixed_pars[11]
    useSigTran = fixed_pars[12]
    useUnrepor = fixed_pars[13]
    useSepTimeForRandAlpha = fixed_pars[14]
    delta_t = fixed_pars[15]
    t_max = fixed_pars[16]
    time_depSoc = fixed_pars[17]

    # Adjacency matrix
    Adj = np.array(fixed_pars[3])

    # Other fit parameters
    R_infU_in = pars[:n_adm]
    T_inc = pars[n_adm]
    R_redU_frac = pars[n_adm + 1:(n_adm + 1) + n_adm]
    b_deR = pars[2 * n_adm + 1:3 * n_adm + 1][0]
    a_deR = pars[3 * n_adm + 1:4 * n_adm + 1][0]
    T_infU = pars[4 * n_adm + 1]
    alpha = pars[4 * n_adm + 2]
    alphaS_us = pars[4 * n_adm + 3]
    R_inf_Ui_in = pars[4 * n_adm + 4:5 * n_adm + 4]
    T_inf_Ui = pars[5 * n_adm + 4]
    p_unr = pars[5 * n_adm + 5]

    # time dependence for measures taken on R
    try:
        if t < t_max:
            t_dep = time_dep(t)
        else:
            t_dep = time_dep(t_max)

        alpha_use = alpha * t_dep

        if useSigmoid:
            R_redU = R_redU_frac * R_infU_in
            R_infU = (R_infU_in - R_redU) / (np.ones(n_adm) + np.exp((t - a_deR) / b_deR)) + R_redU
            # R_infU = (R_infU_in - R_redU) / (np.ones(n_adm) + np.exp(t - a_deR) / b_deR) + R_redU
        else:
            R_infU = R_infU_in * time_depSoc(t)  # timeDep_R(R_infU_in, R_redU_frac, t_dep)

        if useRelativeR:
            for i_tile in range(1, n_adm):
                R_infU[i_tile] = R_infU[0] * R_infU_in[i_tile]

        R_inf_Ui = R_infU
    except:
        print("Interpolation failed for this time: ")
        print(t)

    if constantMobility:
        alpha_use = alpha

        if zeroMobility:
            alpha_use = 0

    if constantR:
        R_infU = R_infU_in
        if useRelativeR:
            for i_tile in range(1, n_adm):
                R_infU[i_tile] = R_infU[0] * R_infU_in[i_tile]

        R_inf_Ui = R_infU_in

    # Multiply with Adj
    M = Adj * alpha_use

    # compartments of each admin area
    s = Y[0 * n_adm:1 * n_adm]
    e = Y[1 * n_adm:2 * n_adm]
    u = Y[2 * n_adm:3 * n_adm]
    i = Y[3 * n_adm:4 * n_adm]
    u_i = Y[4 * n_adm:5 * n_adm]
    u_r = Y[5 * n_adm:6 * n_adm]
    n = s + e + i + u + u_i + u_r

    newInf = np.zeros((s.shape))
    stot = np.zeros((s.shape))
    utot = np.zeros((s.shape))
    u_itot = np.zeros((s.shape))
    ntot = np.zeros((s.shape))
    all_quats = list(np.arange(n_adm))
    for k in all_quats:
        stot[k] = s[k] - np.sum(M[k, :] * s[k])
        utot[k] = np.sum(M[:, k] * u) - np.sum(M[k, :] * u[k]) + u[k]
        u_itot[k] = np.sum(M[:, k] * u_i) - np.sum(M[k, :] * u_i[k]) + u_i[k]
        ntot[k] = np.sum(M[:, k] * n) - np.sum(M[k, :] * n[k]) + n[k]

    if RperTilefix:
        for k in range(0, n_adm):
            newInf[k] = 1. / T_infU * stot[k] / ntot[k] * R_infU[k] * utot[k] + \
                        1. / T_inf_Ui * stot[k] / ntot[k] * R_inf_Ui[k] * u_itot[k]

            for l in range(0, n_adm):
                newInf[k] = newInf[k] + 1. / T_infU * s[k] / ntot[l] * R_infU[l] * utot[l] * M[k, l] + \
                            + 1. / T_inf_Ui * s[k] / ntot[l] * R_inf_Ui[l] * u_itot[l] * M[k, l]
    else:
        for k in range(0, n_adm):
            newInf[k] = 1. / T_infU * stot[k] / ntot[k] * (
                        np.sum(R_infU * M[:, k] * u) + R_infU[k] * (-np.sum(M[k, :] * u[k]) + u[k])) + \
                        1. / T_inf_Ui * stot[k] / ntot[k] * (np.sum(R_inf_Ui * M[:, k] * u_i) + R_inf_Ui[k] * (
                        -np.sum(M[k, :] * u_i[k]) + u_i[k]))
            for l in range(0, n_adm):
                newInf[k] = newInf[k] + 1. / T_infU * s[k] * M[k, l] / ntot[l] * (
                            np.sum(R_infU * M[:, k] * u) + R_infU[l] * (-np.sum(M[l, :] * u[l]) + u[l])) \
                            + 1. / T_inf_Ui * s[k] * M[k, l] / ntot[l] * (
                                        np.sum(R_inf_Ui * M[k, l] * u_i) + R_inf_Ui[l] * (
                                            -np.sum(M[l, :] * u_i[l]) + u_i[l])) \
 \
                    # Susceptibles: Add - diffusion term to E - diffusion term to A
    dsdt = -newInf

    # Exposed - not infectious
    dedt = newInf - np.multiply(1. / T_inc, e)

    # Infectious prior to symptom onset
    dudt = np.multiply(1 / T_inc, e) - np.multiply(1 / T_infU, u)

    # Reported infected - assumed to be isolated
    didt = (1 - p_unr) * np.multiply(1 / T_infU, u)

    # Unreported infected - infectious and not isolated (might know about symptoms, hence different R from U compartment)
    du_idt = p_unr * np.multiply(1 / T_infU, u) - np.multiply(1 / T_inf_Ui, u_i) \
 \
        # Unreported recovered - account for duration of infectious periode
    du_rdt = np.multiply(1 / T_inf_Ui, u_i)

    # output has to have shape
    return np.concatenate((dsdt, dedt, dudt, didt, du_idt, du_rdt))


def ode_model_SEUI_SymMat(Y, t, pars, fixed_pars):
    """ODE model. The order of compartments in Y is s, e, i, r, d, a, ar, i2,
    repeated for the number of considered nodes. If 3 adm1 areas are cosidered,
    than y is: s1, s2, s3, e1, e2, ... and so on.

    Parameters
    ----------
    Y          : vector containing the values of the state variables:
    t          : time vector
    pars       : vector containing the model parameters' current values
    fixed_pars : vector containing the fixed parameters

    Returns
    -------
    list of values of the differentials for each compartment variable at time t
    """
    n_adm = fixed_pars[0]
    n_cmp = fixed_pars[1]
    t_meas = fixed_pars[2]
    c = fixed_pars[4]
    local = fixed_pars[5]
    neigh = fixed_pars[7]
    alp_fix = fixed_pars[8]
    loadFr_dyn = fixed_pars[10]
    time_dep = fixed_pars[11]
    useSigTran = fixed_pars[12]
    useUnrepor = fixed_pars[13]
    useSepTimeForRandAlpha = fixed_pars[14]
    delta_t = fixed_pars[15]
    t_max = fixed_pars[16]

    # Optionally fit adjacency matrix
    if neigh == 'estimate':
        Adj = np.reshape(pars[-n_adm * n_adm:], (n_adm, n_adm))
        AdjSoc = np.reshape(pars[-n_adm * n_adm:], (n_adm, n_adm))
    else:
        Adj = fixed_pars[3]
        AdjSoc = fixed_pars[9]

    # Other fit parameters
    if alp_fix:
        if not local:
            R_infU_in = pars[:n_adm]
            T_inc = pars[n_adm]
            R_redU_frac = pars[n_adm + 1:(n_adm + 1) + n_adm]
            b_deR = pars[2 * n_adm + 1:3 * n_adm + 1][0]
            a_deR = pars[3 * n_adm + 1:4 * n_adm + 1][0]
            T_infU = pars[4 * n_adm + 1]
            alpha = pars[4 * n_adm + 2]
            alphaS_us = pars[4 * n_adm + 3]
            R_inf_Ui_in = pars[4 * n_adm + 4:5 * n_adm + 4]
            T_inf_Ui = pars[5 * n_adm + 4]
            p_unr_in = pars[5 * n_adm + 5]


        elif local:
            R_infU_in = pars[:n_adm]
            T_inc = pars[n_adm]
            R_redU_frac = pars[n_adm + 1:(n_adm + 1) + n_adm]
            b_deR = pars[2 * n_adm + 1:3 * n_adm + 1][0]
            a_deR = pars[3 * n_adm + 1:4 * n_adm + 1][0]
            T_infU = pars[4 * n_adm + 1]
            alpha = pars[4 * n_adm + 2]
            alphaS_us = pars[4 * n_adm + 3]
            R_inf_Ui_in = pars[4 * n_adm + 4:5 * n_adm + 4:]
            T_inf_Ui = pars[5 * n_adm + 4]
            p_unr_in = pars[5 * n_adm + 5]

        try:
            if t < t_max:
                t_dep = time_dep(t)
                t_dep_soc = time_dep_soc(t)
            else:
                t_dep = time_dep(t_max)
                t_dep_soc = time_dep_soc(t_max)

            if useExponentTimeseries:
                alpha_use = alpha * (t_dep ** b_deR)
            else:
                alpha_use = alpha * t_dep
        except:
            print("Interpolation failed for this time: ")
            print(t)

        if useVariable_p_unr:
            p_unr = p_unr_in * sigmoid_p(0.9999 / p_unr_in, R_inf_Ui_in[0], a_deR, b_deR, t)  # ratio_time_dep(t)
        else:
            p_unr = p_unr_in

        # time dependence for measures taken on R
        if useSigmoid:
            raise ('CODE NOT CHECKED!!!')
            if useMultiplicModel:
                alpha_soc = sigmoid_R(1., R_redU_frac[0], a_deR, b_deR, t)
                R_infU = R_infU_in * alpha_soc
            else:
                R_redU = R_redU_frac * R_infU_in
                # R_infU = (R_infU_in - R_redU) / (np.ones(n_adm) + np.exp(t - a_deR) / b_deR) + R_redU
                R_infU = (R_infU_in - R_redU) / (np.ones(n_adm) + np.exp((t - a_deR) / b_deR)) + R_redU

        else:

            if useMultiplicModel:

                if useStretchSocial:
                    stretch = 0.19 * alphaS_us
                else:
                    stretch = 0

                if useHomeReproductive:
                    R_infU_mob = stretchFun(t, 51, stretch) * R_redU_frac[0]
                    R_inf_Ui_mob = stretchFun(t, 51, stretch) * R_redU_frac[0]
                    R_inf_U_base = R_infU_in.copy()
                    R_inf_Ui_base = R_infU_in.copy()
                    R_infU = 0
                else:
                    R_infU = stretchFun(t, 51, stretch) * R_infU_in



            else:
                R_infU = timeDep_R(R_infU_in, R_redU_frac, t_dep)

        if useRelativeR:
            if useHomeReproductive:
                for i_tile in range(1, n_adm):
                    R_inf_U_base[i_tile] = R_inf_U_base[0] * R_infU_in[i_tile]
                    R_inf_Ui_base[i_tile] = R_inf_Ui_base[0] * R_infU_in[i_tile]
            else:
                for i_tile in range(1, n_adm):
                    R_infU[i_tile] = R_infU[0] * R_infU_in[i_tile]

    if constantMobility:
        alpha_use = alpha

        if zeroMobility:
            alpha_use = 0

    if constantR:
        if useHomeReproductive:
            R_infU_mob
            R_inf_Ui_mob
            R_inf_U_base
            R_inf_Ui_base

        else:
            if useRelativeR:
                R_infU[0] = R_infU_in[0].copy()
                for i_tile in range(1, n_adm):
                    R_infU[i_tile] = R_infU_in[0] * R_infU_in[i_tile]
            else:
                R_infU = R_infU_in.copy()

        R_infU = R_infU * fixedSocial

    # Same reproductive number for U and U_i
    R_inf_Ui = R_infU

    # compartments
    s = Y[0 * n_adm:1 * n_adm]
    e = Y[1 * n_adm:2 * n_adm]
    u = Y[2 * n_adm:3 * n_adm]
    i = Y[3 * n_adm:4 * n_adm]
    n = s + e + i + u

    if useUnrepor:
        u_i = Y[4 * n_adm:5 * n_adm]
        u_r = Y[5 * n_adm:6 * n_adm]
        n = n + u_i + u_r

        if useReortingDelay:
            i = Y[6 * n_adm:7 * n_adm]
            r = Y[3 * n_adm:4 * n_adm]
            n = n + i
            T_delay = pars[n_adm]
            T_inc = 4

        # Susceptibles: Add - diffusion term to E - diffusion term to A
        if useMultiplicModel:
            if useHomeReproductive:

                dsdt = - np.multiply(alpha_use * s * R_infU_mob / (n), np.dot(Adj, np.multiply(1. / T_infU, u))) \
                       - np.multiply(alpha_use * s * R_inf_Ui_mob / (n), np.dot(Adj, np.multiply(1. / T_inf_Ui, u_i))) \
                       - s * R_inf_Ui_base * u_i / (n * T_inf_Ui) \
                       - s * R_inf_U_base * u / (n * T_infU)

                # Exposed - not infectious
                dedt = - np.multiply(1 / T_inc, e) \
                       + np.multiply(alpha_use * s * R_infU_mob / (n), np.dot(Adj, np.multiply(1. / T_infU, u))) \
                       + np.multiply(alpha_use * s * R_inf_Ui_mob / (n), np.dot(Adj, np.multiply(1. / T_inf_Ui, u_i))) \
                       + s * R_inf_Ui_base * u_i / (n * T_inf_Ui) \
                       + s * R_inf_U_base * u / (n * T_infU)
            else:
                dsdt = - np.multiply(alpha_use * s * R_infU / (n), np.dot(Adj, np.multiply(1. / T_infU, u))) \
                       - np.multiply(alpha_use * s * R_inf_Ui / (n), np.dot(Adj, np.multiply(1. / T_inf_Ui, u_i)))

                # Exposed - not infectious
                dedt = - np.multiply(1 / T_inc, e) \
                       + np.multiply(alpha_use * s * R_infU / (n), np.dot(Adj, np.multiply(1. / T_infU, u))) \
                       + np.multiply(alpha_use * s * R_inf_Ui / (n), np.dot(Adj, np.multiply(1. / T_inf_Ui, u_i)))


        else:

            dsdt = - np.multiply(np.multiply(R_infU / T_infU, u), s / n) \
                   - c * (np.multiply(alpha_use * s * R_infU / (n), np.dot(Adj, np.multiply(1. / T_infU, u)))) \
                   - np.multiply(np.multiply(R_inf_Ui / T_inf_Ui, u_i), s / n) \
                   - c * (np.multiply(alpha_use * s * R_inf_Ui / (n), np.dot(Adj, np.multiply(1. / T_inf_Ui, u_i)))) \
 \
                # Exposed - not infectious
            dedt = np.multiply(np.multiply(R_infU / T_infU, u), s / n) \
                   - np.multiply(1 / T_inc, e) \
                   + np.multiply(np.multiply(R_inf_Ui / T_inf_Ui, u_i), s / n) \
                   + c * (np.multiply(alpha_use * s * R_infU / (n), np.dot(Adj, np.multiply(1. / T_infU, u)))) \
                   + c * (np.multiply(alpha_use * s * R_inf_Ui / (n), np.dot(Adj, np.multiply(1. / T_inf_Ui, u_i)))) \
 \
                # Infectious prior to symptom onset
        dudt = np.multiply(1 / T_inc, e) - np.multiply(1 / T_infU, u)

        # Reported infected - assumed to be isolated
        if useReortingDelay:
            didt = (1 - p_unr) * np.multiply(1 / T_infU, u) - np.multiply(1 / T_delay, i)
            drdt = np.multiply(1 / T_delay, i)
        else:
            didt = (1 - p_unr) * np.multiply(1 / T_infU, u)

        if useNoUi:

            # Unreported infected - infectious and not isolated (might know about symptoms, hence different R from U compartment)
            du_idt = np.zeros(u.shape)

            # Unreported recovered - account for duration of infectious periode
            du_rdt = p_unr * np.multiply(1 / T_infU, u)
        else:
            # Unreported infected - infectious and not isolated (might know about symptoms, hence different R from U compartment)
            du_idt = p_unr * np.multiply(1 / T_infU, u) - np.multiply(1 / T_inf_Ui, u_i) \
 \
                # Unreported recovered - account for duration of infectious periode
            du_rdt = np.multiply(1 / T_inf_Ui, u_i)

    else:

        du_idt = 0
        du_rdt = 0

        # Susceptibles: Add - diffusion term to E - diffusion term to A
        dsdt = - np.multiply(np.multiply(R_infU / T_infU, u), s / n) \
               - c * (np.multiply(alpha_use * s / (n + np.sum(Adj, axis=0)), \
                                  np.dot(Adj, np.multiply(R_infU / T_infU, np.multiply(s, u) / n)))) \
               - c * (np.multiply(alphaS_us * s / (n + np.sum(AdjSoc, axis=0)), \
                                  np.dot(AdjSoc, np.multiply(R_infU / T_infU, np.multiply(s, u) / n))))

        # Exposed - not infectious
        dedt = np.multiply(np.multiply(R_infU / T_infU, u), s / n) \
               - np.multiply(1 / T_inc, e)

        # Infectious prior to symptom onset
        dudt = np.multiply(1 / T_inc, e) - np.multiply(1 / T_infU, u) \
               + c * (np.multiply(alpha_use * s / (n + np.sum(Adj, axis=0)), \
                                  np.dot(Adj, np.multiply(R_infU / T_infU, np.multiply(s, u) / n)))) \
               + c * (np.multiply(alphaS_us * s / (n + np.sum(AdjSoc, axis=0)), \
                                  np.dot(AdjSoc, np.multiply(R_infU / T_infU, np.multiply(s, u) / n))))

        # Reported infected
        didt = np.multiply(1 / T_infU, u)

    # output has to have shape
    if useReortingDelay:
        output = np.concatenate((dsdt, dedt, dudt, drdt, du_idt, du_rdt, didt))
    else:
        output = np.concatenate((dsdt, dedt, dudt, didt, du_idt, du_rdt))

    return output


def ode_model_SEUI_parallel(Y, t, pars, fixed_pars):
    """ODE model. The order of compartments in Y is s, e, i, r, d, a, ar, i2,
    repeated for the number of considered nodes. If 3 adm1 areas are cosidered,
    than y is: s1, s2, s3, e1, e2, ... and so on.

    Parameters
    ----------
    Y          : vector containing the values of the state variables:
    t          : time vector
    pars       : vector containing the model parameters' current values
    fixed_pars : vector containing the fixed parameters

    Returns
    -------
    list of values of the differentials for each compartment variable at time t
    """
    n_adm = fixed_pars[0]
    c = fixed_pars[4]
    local = fixed_pars[5]
    neigh = fixed_pars[7]
    alp_fix = fixed_pars[8]
    loadFr_dyn = fixed_pars[10]
    time_dep = fixed_pars[11]
    useSigTran = fixed_pars[12]
    useUnrepor = fixed_pars[13]
    delta_t = fixed_pars[15]
    t_max = fixed_pars[16]
    AdjSoc = 0

    n_separations = fixed_pars[17]
    n_parameters = fixed_pars[-1]

    # Parameters shared between separations
    alpha = pars[4 * n_adm + 2]
    alphaS_us = pars[4 * n_adm + 3]
    T_infU = pars[4 * n_adm + 1]
    T_inf_Ui = pars[5 * n_adm + 4]
    T_inc = pars[n_adm]
    p_unr_in = pars[5 * n_adm + 5]

    for i_ind in range(0, n_separations):

        # Optionally fit adjacency matrix
        if i_ind == 0:
            # Other fit parameters specific to each separation
            b_deR = pars[2 * n_adm + 1:3 * n_adm + 1][0]
            a_deR = pars[3 * n_adm + 1:4 * n_adm + 1][0]
            R_infU_in = pars[:n_adm]
            R_redU_frac = pars[n_adm + 1:(n_adm + 1) + n_adm]
            Adj = fixed_pars[3]
        else:

            # Other fit parameters specific to each separation
            b_deR = pars[(i_ind - 1) * 4 * n_adm + n_parameters:(i_ind - 1) * 4 * n_adm + n_parameters + n_adm][0]
            a_deR = \
            pars[(i_ind - 1) * 4 * n_adm + n_parameters + n_adm:(i_ind - 1) * 4 * n_adm + n_parameters + 2 * n_adm][0]
            R_infU_in = pars[(i_ind - 1) * 4 * n_adm + n_parameters + 2 * n_adm:(
                                                                                            i_ind - 1) * 4 * n_adm + n_parameters + 3 * n_adm]
            R_redU_frac = pars[(i_ind - 1) * 4 * n_adm + n_parameters + 3 * n_adm:(
                                                                                              i_ind - 1) * 4 * n_adm + n_parameters + 4 * n_adm]
            Adj = fixed_pars[18 + (i_ind - 1) * 2]

            if useSame_ab:
                b_deR = pars[2 * n_adm + 1:3 * n_adm + 1]
                a_deR = pars[3 * n_adm + 1:4 * n_adm + 1]

        # Time dependence of mobility
        if t < t_max:
            t_dep = time_dep(t)
        else:
            t_dep = time_dep(t_max)
        alpha_use = alpha * t_dep

        # Unreported cases
        if useVariable_p_unr:
            p_unr = p_unr_in * sigmoid_p(0.9999 / p_unr_in, R_inf_Ui_in[0], a_deR, b_deR, t)  # ratio_time_dep(t)
        else:
            p_unr = p_unr_in

        # time dependence for measures taken on R
        if useSigmoid:
            raise ('CODE NOT CHECKED!!!')
            if useMultiplicModel:
                alpha_soc = sigmoid_R(1., R_redU_frac[0], a_deR, b_deR, t)
                R_infU = R_infU_in * alpha_soc
            else:
                R_redU = R_redU_frac * R_infU_in
                # R_infU = (R_infU_in - R_redU) / (np.ones(n_adm) + np.exp(t - a_deR) / b_deR) + R_redU
                R_infU = (R_infU_in - R_redU) / (np.ones(n_adm) + np.exp((t - a_deR) / b_deR)) + R_redU
        else:
            if useMultiplicModel:
                if useStretchSocial:
                    stretch = 0.19 * alphaS_us
                else:
                    stretch = 0

                if useHomeReproductive:
                    R_infU_mob = stretchFun(t, 51, stretch) * R_redU_frac[0]
                    R_inf_Ui_mob = stretchFun(t, 51, stretch) * R_redU_frac[0]
                    R_inf_U_base = R_infU_in.copy()
                    R_inf_Ui_base = R_infU_in.copy()
                    R_infU = 0
                else:
                    R_infU = stretchFun(t, 51, stretch) * R_infU_in
            else:
                R_infU = timeDep_R(R_infU_in, R_redU_frac, t_dep)

        # Relative reproductive number
        if useRelativeR:
            if useHomeReproductive:
                for i_tile in range(1, n_adm):
                    R_inf_U_base[i_tile] = R_inf_U_base[0] * R_infU_in[i_tile]
                    R_inf_Ui_base[i_tile] = R_inf_Ui_base[0] * R_infU_in[i_tile]
            else:
                for i_tile in range(1, n_adm):
                    R_infU[i_tile] = R_infU[0] * R_infU_in[i_tile]

        # Constant mobility
        if constantMobility:
            alpha_use = alpha
            if zeroMobility:
                alpha_use = 0

        # Constant social interaction
        if constantR:
            if useHomeReproductive:
                R_infU_mob
                R_inf_Ui_mob
                R_inf_U_base
                R_inf_Ui_base

            else:
                if useRelativeR:
                    R_infU[0] = R_infU_in[0].copy()
                    for i_tile in range(1, n_adm):
                        R_infU[i_tile] = R_infU_in[0] * R_infU_in[i_tile]
                else:
                    R_infU = R_infU_in.copy()

        # Same reproductive number for U and U_i
        R_inf_Ui = R_infU

        # compartments
        s = Y[i_ind * 6 * n_adm + 0 * n_adm:i_ind * 6 * n_adm + 1 * n_adm]
        e = Y[i_ind * 6 * n_adm + 1 * n_adm:i_ind * 6 * n_adm + 2 * n_adm]
        u = Y[i_ind * 6 * n_adm + 2 * n_adm:i_ind * 6 * n_adm + 3 * n_adm]
        i = Y[i_ind * 6 * n_adm + 3 * n_adm:i_ind * 6 * n_adm + 4 * n_adm]
        u_i = Y[i_ind * 6 * n_adm + 4 * n_adm:i_ind * 6 * n_adm + 5 * n_adm]
        u_r = Y[i_ind * 6 * n_adm + 5 * n_adm:i_ind * 6 * n_adm + 6 * n_adm]
        n = s + e + u + i + u_i + u_r

        if useReortingDelay:
            raise ('Option useReortingDelay not supported!')

        if useHomeReproductive:
            raise ('Option useHomeReproductive not supported!')

        if useMultiplicModel:
            dsdt = - np.multiply(alpha_use * s * R_infU / (n), np.dot(Adj, np.multiply(1. / T_infU, u))) \
                   - np.multiply(alpha_use * s * R_inf_Ui / (n), np.dot(Adj, np.multiply(1. / T_inf_Ui, u_i)))

            # Exposed - not infectious
            dedt = - np.multiply(1 / T_inc, e) \
                   + np.multiply(alpha_use * s * R_infU / (n), np.dot(Adj, np.multiply(1. / T_infU, u))) \
                   + np.multiply(alpha_use * s * R_inf_Ui / (n), np.dot(Adj, np.multiply(1. / T_inf_Ui, u_i)))


        else:
            raise ('Have to use multiplicative model!')

            # Susceptibles: Add - diffusion term to E - diffusion term to A
            dsdt = - np.multiply(np.multiply(R_infU / T_infU, u), s / n) \
                   - c * (np.multiply(alpha_use * s * R_infU / (n), np.dot(Adj, np.multiply(1. / T_infU, u)))) \
                   - np.multiply(np.multiply(R_inf_Ui / T_inf_Ui, u_i), s / n) \
                   - c * (np.multiply(alpha_use * s * R_inf_Ui / (n), np.dot(Adj, np.multiply(1. / T_inf_Ui, u_i)))) \
 \
                # Exposed - not infectious
            dedt = np.multiply(np.multiply(R_infU / T_infU, u), s / n) \
                   - np.multiply(1 / T_inc, e) \
                   + np.multiply(np.multiply(R_inf_Ui / T_inf_Ui, u_i), s / n) \
                   + c * (np.multiply(alpha_use * s * R_infU / (n), np.dot(Adj, np.multiply(1. / T_infU, u)))) \
                   + c * (np.multiply(alpha_use * s * R_inf_Ui / (n), np.dot(Adj, np.multiply(1. / T_inf_Ui, u_i)))) \
 \
                # Infectious prior to symptom onset
        dudt = np.multiply(1 / T_inc, e) - np.multiply(1 / T_infU, u)

        # Reported infected - assumed to be isolated
        didt = (1 - p_unr) * np.multiply(1 / T_infU, u)
        if np.any(didt < 0):
            print('SOMETHING IS WRONG!!!')

        if useNoUi:

            # Unreported infected - infectious and not isolated (might know about symptoms, hence different R from U compartment)
            du_idt = np.zeros(u.shape)

            # Unreported recovered - account for duration of infectious periode
            du_rdt = p_unr * np.multiply(1 / T_infU, u)
        else:
            # Unreported infected - infectious and not isolated (might know about symptoms, hence different R from U compartment)
            du_idt = p_unr * np.multiply(1 / T_infU, u) - np.multiply(1 / T_inf_Ui, u_i)

            # Unreported recovered - account for duration of infectious periode
            du_rdt = np.multiply(1 / T_inf_Ui, u_i)

        # output has to have shape
        if i_ind == 0:
            out = np.concatenate((dsdt, dedt, dudt, didt, du_idt, du_rdt))
        else:
            out = np.concatenate((out, dsdt, dedt, dudt, didt, du_idt, du_rdt))

    return out


def dofit_SEUI(par_list, data_train, bnds, fixed_pars, adm0):
    result = \
        minimize(lambda var1, var2: residual_SEUI(var1, var2, fixed_pars, adm0), \
                 par_list, \
                 args=(data_train), \
                 method='L-BFGS-B', \
                 bounds=bnds, \
                 # tol=1e-5, \
                 options={'gtol': 1e-8, 'ftol': 1e-8, 'disp': True, 'maxiter': 200})  # 200

    print(' ')
    print('n_exp_0 = ' + str(result.x[0]))
    print('n_inf_0 = ' + str(result.x[2]))
    print('n_und_0 = ' + str(result.x[1]))
    return result


def residual_SEUI_parallel(pars, data, fixed_pars, adm0):
    '''Computes residuals of the data with respect to fitted model

    Parameters
    -----------
    pars       : initial conditions and parameter values to be optimized
    data       : array containing the training data
    fixed_pars : fixed parameters
    adm0       : iso3 pcode of country of interest

    Returns
    -----------
    sum of the infected's and deceased's residuals
    '''
    n_adm = fixed_pars[0]
    n_cmp = fixed_pars[1]
    t = data[0][0]
    n_separations = fixed_pars[17]

    # Solution with given parameter set
    yfit = solution_SEUI_parallel(t, pars, fixed_pars)
    yfit = yfit.T

    res_inf = 0
    for i in range(0, n_separations):
        yfit_inf = yfit[i * n_cmp * n_adm + 3 * n_adm:i * n_cmp * n_adm + 4 * n_adm]
        yfit_cuminf = yfit_inf

        n_cuminf = data[i][1:n_adm + 1, :]
        # res_inf = np.mean((yfit_cuminf - n_cuminf) ** 2)

        inds = np.logical_and(~np.isinf(np.log(n_cuminf)),
                              ~np.isinf(np.log(yfit_cuminf)))
        weights = np.ones(yfit_cuminf[inds].shape)
        weights[n_cuminf[inds] < 15] = 0.0
        this_res_inf = np.sum(weights * (np.log(yfit_cuminf[inds]) - np.log(n_cuminf[inds])) ** 2)
        res_inf = res_inf + this_res_inf

    return res_inf


def residual_SEUI(pars, data, fixed_pars, adm0):
    '''Computes residuals of the data with respect to fitted model

    Parameters
    -----------
    pars       : initial conditions and parameter values to be optimized
    data       : array containing the training data
    fixed_pars : fixed parameters
    adm0       : iso3 pcode of country of interest

    Returns
    -----------
    sum of the infected's and deceased's residuals
    '''
    n_adm = fixed_pars[0]
    n_cmp = fixed_pars[1]
    t = data[0]
    n_cuminf = data[1:n_adm + 1, :]
    # n_fat    = data[n_adm+1:]

    # Solution with given parameter set
    yfit = solution_SEUI(t, pars, fixed_pars)
    yfit = yfit.T

    yfit_inf = yfit[3 * n_adm:4 * n_adm]
    yfit_cuminf = yfit_inf

    # res_inf = np.mean((yfit_cuminf - n_cuminf) ** 2)

    if multiplyCaseNUmbers:
        n_cuminf = n_cuminf * 12.5

    if useGradient:
        res_inf = 0
        for i in range(0, n_adm):
            inds = np.logical_and(~np.isinf(np.log(np.gradient(n_cuminf[i, :]))),
                                  ~np.isinf(np.log(np.gradient(yfit_cuminf[i, :]))))
            yfit_cuminf_use = yfit_cuminf[i, inds]
            n_cuminf_use = n_cuminf[i, inds]
            res_inf = res_inf + np.sum((np.log(np.gradient(yfit_cuminf_use)) - np.log(np.gradient(n_cuminf_use))) ** 2)
    else:
        inds = np.logical_and(~np.isinf(np.log(n_cuminf)),
                              ~np.isinf(np.log(yfit_cuminf)))
        weights = np.ones(yfit_cuminf[inds].shape)
        weights[n_cuminf[inds] < 15] = 0.0
        res_inf = np.sum(weights * (np.log(yfit_cuminf[inds]) - np.log(n_cuminf[inds])) ** 2)

    return res_inf


def solution_SEUI(t, par_list, fixed_pars):
    '''Numerical ODE solution for given set of parameters.

    Parameters
    ----------
    t          : time points vector
    par_list   : initial conditions and AFTER the initial parameters values.
    fixed_pars : fixed parameters

    Returns
    -------
    numerical solution of the ODE system function
    '''
    n_adm = fixed_pars[0]
    n_cmp = fixed_pars[1]
    s_0 = fixed_pars[6]
    y0 = np.concatenate((s_0, par_list[:((n_cmp - 1) * n_adm)]))
    pars = par_list[(n_cmp - 1) * n_adm:]

    if useSymMat:
        sol = odeint(lambda a, b, c: ode_model_SEUI_SymMat(a, b, c, fixed_pars), y0, t, args=(pars,))
    else:
        sol = odeint(lambda a, b, c: ode_model_SEUI(a, b, c, fixed_pars), y0, t, args=(pars,))

    return sol


# Uncertainty
def loadUncertData(n_uncert, quat, uncData_id, t_trn):
    data_in = [np.zeros((len(quat) + 1, len(t_trn))) for i in range(n_uncert + 1)]
    counter = 0
    for j, q in enumerate(quat):
        df_data = pd.read_csv(os.path.join(wd, 'Data', str(q) + '_data_' + uncData_id + '.csv'))
        df_data = df_data.drop('Unnamed: 0', axis=1)
        # df_time = pd.read_csv(os.path.join(wd,'Data',str(q)+'_time_'+uncData_id+'.csv'))
        for i in range(0, len(data_in)):
            if counter == 0:
                data_in[i][0, :] = t_trn
            data_in[i][j + 1, :] = df_data.loc[i, :].values[t_trn.astype(int)]
        counter = counter + 1

    return data_in


def uncertFit_loaded(input):
    data_in = input[0]
    t_trn = input[1]
    model = input[2]
    par_list = input[3]
    bnds = input[4]
    fixed_pars = input[5]
    adm0 = input[6]
    t_tst = input[7]
    neigh = input[8]
    alpha_fix = input[9]
    index = input[10]

    # fit model
    result_i, t_i, fit_i = fit_general(model, par_list, data_in[index], bnds, fixed_pars, adm0, t_trn,
                                       t_tst, neigh, alpha_fix)
    out = [result_i, data_in[index], fit_i]
    return out


def uncertFit(input):
    data_trn = input[0]
    t_trn = input[1]
    model = input[2]
    par_list = input[3]
    bnds = input[4]
    fixed_pars = input[5]
    adm0 = input[6]
    t_tst = input[7]
    neigh = input[8]
    alpha_fix = input[9]
    inds_s10 = input[10]
    rmse = input[11]
    mean_points = input[12]
    rand_seed = input[13]
    n_adm = fixed_pars[0]

    # Only accept runs with a minimum goodness of fit
    acceptableFit = 0
    counter = 0
    while acceptableFit < 1 and counter < 1:

        np.random.seed(rand_seed + counter * 50)

        # Randomly shift data for each quarter
        data_new = data_trn.copy()
        plt.figure()
        for i in range(0, n_adm):

            # Get absolute mean point
            mean_points_thisQuarter = mean_points[i, :]
            data = np.multiply(mean_points_thisQuarter, np.random.normal(1, rmse, size=mean_points_thisQuarter.shape))
            data[inds_s10[i]] = np.multiply(mean_points_thisQuarter[inds_s10[i]],
                                            np.random.normal(1, rmse * 2,
                                                             size=mean_points_thisQuarter[inds_s10[i]].shape))
            data[data < 0] = 0

            # Convert to cummulative numbers and 7-day average
            if use7DayAV:
                if use7DayAV:
                    y_inf_new_7av = np.zeros(data.shape)
                    for d in range(0, len(data_new[0, :])):
                        ind = np.arange(d - 3, d + 4)
                        ind_use = ind[np.logical_and(ind >= 0, ind < len(data_new[0, :]))]
                        y_inf_new_7av[d] = np.mean(np.array(data)[ind_use])
                    y_inf_7av = np.cumsum(y_inf_new_7av)

                data_use = y_inf_7av
            else:
                data_use = data

            data_new[1 + i, :] = data_use

            # plt.plot(data_new[0,:],data_new[i+1,:],'-')
            # plt.plot(data_new[0, :], data_trn[i + 1, :], '-')

        # Fit model
        result_i, t_i, fit_i = fit_general(model, par_list, data_new, bnds, fixed_pars, adm0, t_trn,
                                           t_tst, neigh, alpha_fix)

        # Calculate goodness of fit
        yfit_inf = fit_i[3 * n_adm:4 * n_adm]
        r2 = [[] for i in range(0, n_adm)]
        for i in range(0, n_adm):
            if useGradient:
                r2[i] = r2_score(np.gradient(data_new[i + 1, :]), np.gradient(yfit_inf[i, :]))
            else:
                r2[i] = r2_score(data_new[i + 1, :], yfit_inf[i, :])

        # Check
        success_lst = [i for i in r2 if i > r2gof_threshRsq]
        if len(success_lst) == n_adm:

            # Accept fit
            acceptableFit = 1
        else:

            # Change random seed and repeat fit
            counter = counter + 1

    out = [result_i, data_new, fit_i, acceptableFit, r2]
    return out


# Save and plot after the fit
def save_data_general(model, adm0, quat, n_adm, core_name, fit, t_trn, t_tst, data_trn, data_tst, pop, start, end,
                      result, fixed_pars, newdir=[], useUnrepor=True):
    '''Saves the data .csv files and figures per adm area. Prints R2.

    Parameters
    ----------
    adm0      : iso3 string of the country of interest
    adm1s     : vector with the analysed adm codes
    n_adm     : number of analyzed admin areas
    core_name : piece of the name of the csv file
    fit       : fit results matrix
    t_trn     : time points used for training
    t_tst     : time points reseved for testing
    data_trn  : data used for training
    data_tst  : data reseved for testing
    pop       : total population of the administrative area of interest
    start     : start date of analysis
    end       : end date of analysis
    '''

    if model == 'SEAIRD':
        save_data(adm0, quat, n_adm, core_name, fit, t_trn, t_tst, data_trn, data_tst, \
                  pop, start, end, result)
    elif model == 'SEUI':
        save_data_SEUI(adm0, quat, n_adm, core_name, fit, t_trn, t_tst, data_trn, data_tst, \
                       pop, start, end, result, fixed_pars, newdir, useUnrepor)
    else:
        raise ('Invalid ODE model')


def plot_ILGE(fit, t, n_adm, data_trn, data_tst, pop, result, model):
    if model == 'SEAIRD':
        yfit_inf = fit[2 * n_adm:3 * n_adm]
        yfit_rec = fit[3 * n_adm:4 * n_adm]
        yfit_fat = fit[4 * n_adm:5 * n_adm]
        yfit_u = fit[7 * n_adm:8 * n_adm]
        yfit_cuminf = yfit_inf + yfit_rec + yfit_fat

        n_inf_data = data_trn[2] + data_tst[2]
        r2_inf = r2_score(n_inf_data, np.squeeze(yfit_cuminf))
        print('R2: ', r2_inf)

        plot2(t, n_inf_data, np.squeeze(yfit_cuminf), np.squeeze(yfit_inf), np.squeeze(fit[5 * n_adm:6 * n_adm]),
              np.squeeze(yfit_rec), np.squeeze(yfit_fat), pop, result)
    elif model == 'SEUI':
        yfit_inf = fit[3 * n_adm:4 * n_adm]
        yfit_a = fit[2 * n_adm:3 * n_adm]
        yfit_e = fit[1 * n_adm:2 * n_adm]

        n_inf_data = np.array(list(data_trn[1]) + list(data_tst[1]))
        r2_inf = r2_score(n_inf_data, np.squeeze(yfit_inf))
        print('R2: ', r2_inf)

        plot_SEUI(t, n_inf_data, np.squeeze(yfit_inf), np.squeeze(yfit_a), np.squeeze(yfit_e), pop, result)
    else:
        raise ('Invalid ODE model')
    return 0


def save_pars_ILGE(adm0, core_name, fitted, fixed, bnds, data_in):
    '''Saves model parameters in pickle format

    Parameters
    ----------
    adm0      : iso3 string of country of interest
    core_name : core name of file to be saved
    fitted    : optimized parameters
    fixed     : fixed parameters
    '''

    with open(adm0 + core_name + '_fitted.pkl', 'wb') as f:
        pickle.dump(fitted, f, pickle.HIGHEST_PROTOCOL)
    with open(adm0 + core_name + '_fixed.pkl', 'wb') as f:
        pickle.dump(fixed, f, pickle.HIGHEST_PROTOCOL)
    with open(adm0 + core_name + '_bounds.pkl', 'wb') as f:
        pickle.dump(bnds, f, pickle.HIGHEST_PROTOCOL)
    with open(adm0 + core_name + '_datain.pkl', 'wb') as f:
        pickle.dump(data_in, f, pickle.HIGHEST_PROTOCOL)


def save_data_SEUI(adm0, adm1s, n_adm, core_name, fit, t_trn, t_tst, data_trn, data_tst, pop, start, end, result,
                   fixed_pars, newdir, useUnrepor):
    '''Saves the data .csv files and figures per adm area. Prints R2.

    Parameters
    ----------
    adm0      : iso3 string of the country of interest
    adm1s     : vector with the analysed adm codes
    n_adm     : number of analyzed admin areas
    core_name : piece of the name of the csv file
    fit       : fit results matrix
    t_trn     : time points used for training
    t_tst     : time points reseved for testing
    data_trn  : data used for training
    data_tst  : data reseved for testing
    pop       : total population of the administrative area of interest
    start     : start date of analysis
    end       : end date of analysis
    '''

    local = fixed_pars[5]
    neigh = fixed_pars[7]
    alp_fix = fixed_pars[8]
    c = fixed_pars[4]
    loadFr_dyn = fixed_pars[10]
    time_dep = fixed_pars[11]
    useSigTran = fixed_pars[12]

    # training data
    cuminf_trn = data_trn[1:n_adm + 1, :].reshape(n_adm, len(t_trn))

    # test data
    cuminf_tst = data_tst[1:n_adm + 1, :].reshape(n_adm, len(t_tst))

    # date labels for time axis
    dates = np.array(pd.date_range(start, end + dt.timedelta(1)))

    if ('all' in adm1s):
        adm1s = np.array([adm0])
    else:

        # plot adjacency matrix
        if c > 0:
            try:
                if neigh == 'estimate':
                    Adj = np.reshape(result[0].x[-n_adm * n_adm:], (n_adm, n_adm))
                else:
                    Adj = fixed_pars[3]

                # plot heatmap
                fig, ax = plt.subplots()
                im = ax.imshow(Adj)

                # We want to show all ticks...
                ax.set_xticks(np.arange(len(adm1s)))
                ax.set_yticks(np.arange(len(adm1s)))
                # ... and label them with the respective list entries
                ax.set_xticklabels(adm1s)
                ax.set_yticklabels(adm1s)

                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                         rotation_mode="anchor")

                # Loop over data dimensions and create text annotations.
                for i in range(len(adm1s)):
                    for j in range(len(adm1s)):
                        try:
                            ax.text(j, i, Adj[i, j], ha="center", va="center", color="w")
                        except:
                            print('No text')

                ax.set_title("Adjacency Matrix")
                fig.tight_layout()
                plt.show()
                file = os.path.join(newdir, 'figures', 'adj_mat' + core_name + '.png')
                plt.savefig(file)
            except:
                print('Figure adjacency matrix failed!')
        else:
            print('No adjacency matrix required!')

    for adm in adm1s:
        idx = np.where(np.array(adm1s) == adm)[0][0]

        mdl_sus = fit[0][0 * n_adm + idx]
        mdl_exp = fit[0][1 * n_adm + idx]
        mdl_unr = fit[0][2 * n_adm + idx]
        mdl_inf = fit[0][3 * n_adm + idx]
        mdl_cuminf = mdl_inf

        if useUnrepor:
            mdl_u_i = fit[0][4 * n_adm + idx]
            mdl_u_r = fit[0][5 * n_adm + idx]

        t = np.concatenate((t_trn, t_tst))

        os.chdir(os.path.join(newdir, 'fitting'))
        save_csv(t, mdl_inf, adm, '_n_infected', core_name)
        save_csv(t, mdl_exp, adm, '_n_exposed', core_name)
        save_csv(t, mdl_sus, adm, '_n_susceptible', core_name)
        save_csv(t, mdl_unr, adm, '_n_asyminfected', core_name)
        if useUnrepor:
            save_csv(t, mdl_u_i, adm, '_n_unr_inf', core_name)
            save_csv(t, mdl_u_r, adm, '_n_unr_rec', core_name)

        if alp_fix:
            RU = np.zeros((56, len(result)))
            RU_i = np.zeros((56, len(result)))
            alpha_use = np.zeros((56, len(result)))
            for l in range(0, len(result)):
                # if loadFr_dyn:
                #     R_initU = result[l].x[5 * n_adm:6 * n_adm]
                #     R_initU_i = result[l].x[9*n_adm+4:10*n_adm+4:]
                #     RU[:, l] = [R_initU[idx] *time_dep(t) for t in range(1, 57)]
                #     RU_i[:, l] = [R_initU_i[idx] * time_dep(t) for t in range(1, 57)]
                # else:

                if useSigTran:

                    if useUnrepor:
                        a = result[l].x[8 * n_adm + 1:9 * n_adm + 1]
                        b = result[l].x[7 * n_adm + 1:8 * n_adm + 1]
                        R_initU = result[l].x[5 * n_adm:6 * n_adm]
                        R_initU_i = result[l].x[9 * n_adm + 4:10 * n_adm + 4]
                        R_redU = result[l].x[6 * n_adm + 1:(6 * n_adm + 1) + n_adm]
                        alpha = result[l].x[9 * n_adm + 2]


                    else:
                        a = result[l].x[8 * n_adm + 1:9 * n_adm + 1]
                        b = result[l].x[7 * n_adm + 1:8 * n_adm + 1]
                        R_initU = result[l].x[5 * n_adm:6 * n_adm]
                        R_redU = result[l].x[6 * n_adm + 1:7 * n_adm + 1]
                        alpha = result[l].x[9 * n_adm + 2]

                    if loadFr_dyn:
                        alpha_use[:, l] = [alpha * time_dep(t) for t in range(1, 57)]

                    RU[:, l] = [sigmoid_R(R_initU[idx], R_redU[idx], a[idx], b[idx], t) for t in range(1, 57)]
                    RU_i[:, l] = [sigmoid_R_withSlope(R_initU_i[idx], R_redU[idx], a[idx], b[idx], t) for t in
                                  range(1, 57)]


                else:
                    a = result[l].x[8 * n_adm + 1:9 * n_adm + 1]
                    b = result[l].x[7 * n_adm + 1:8 * n_adm + 1]
                    R_initU = result[l].x[5 * n_adm:6 * n_adm]
                    R_redU = result[l].x[6 * n_adm + 1:7 * n_adm + 1]

                    RU[:, l] = [sigmoid_R(R_initU[idx], R_redU[idx], a[idx], b[idx], t) for t in range(1, 57)]

            df_RU = pd.DataFrame(data=RU, index=np.arange(1, 57))
            df_RU.to_csv('_R_U' + core_name + 'quarter' + str(adm) + '.csv')

            if useUnrepor:
                df_RU_i = pd.DataFrame(data=RU_i, index=np.arange(1, 57))
                df_RU_i.to_csv('_R_U_i' + core_name + 'quarter' + str(adm) + '.csv')

            if loadFr_dyn:
                df_alpha = pd.DataFrame(data=alpha_use, index=np.arange(1, 57))
                df_alpha.to_csv('_alpha' + core_name + 'quarter' + str(adm) + '.csv')


        else:
            raise ('This option is not fully implemented yet!')
            a = result[0].x[6 * n_adm + 1:7 * n_adm + 1]
            b = result[0].x[5 * n_adm + 1:6 * n_adm + 1]
            alpha_in = result[0].x[3 * n_adm:4 * n_adm]
            alpha_red = result[0].x[4 * n_adm + 1:5 * n_adm + 1]

            alpha = [sigmoid_R(alpha_in[idx], alpha_red[idx], a[idx], b[idx], t) for t in range(1, 57)]

            save_csv(np.arange(56), np.array(alpha), adm, '_alpha_', core_name + 'quarter' + str(adm))

        os.chdir(wd)
        real_cuminf_trn = cuminf_trn[idx]
        real_cuminf_tst = cuminf_tst[idx]

        os.chdir(os.path.join(newdir, 'original'))
        save_csv(t_trn, real_cuminf_trn, adm, '_n_confirmed', core_name + '_trn')
        save_csv(t_tst, real_cuminf_tst, adm, '_n_confirmed', core_name + '_tst')
        os.chdir(wd)

        # R2
        r2_inf_trn = r2_score(real_cuminf_trn, mdl_cuminf[:len(t_trn)])
        print('\nR2 infected train: %.3f' % (r2_inf_trn))

        if len(t_tst > 0):
            r2_inf_tst = r2_score(real_cuminf_tst, mdl_cuminf[len(t_trn):])
            print('\nR2 infected test: %.3f' % (r2_inf_tst))
        else:
            r2_inf_tst = np.nan
            print('\n Not enough data to calculate R2 infected test')

        os.chdir(os.path.join(newdir, 'figures'))
        yfit_inf = fit[0][3 * n_adm:4 * n_adm]
        yfit_a = fit[0][2 * n_adm:3 * n_adm]
        yfit_e = fit[0][1 * n_adm:2 * n_adm]

        n_inf_data = np.array(list(data_trn[idx + 1, :]) + list(data_tst[idx + 1, :]))
        r2_inf = r2_score(n_inf_data, np.squeeze(yfit_inf[idx, :]))
        print('R2 trn+tst: ', r2_inf)

        plot_SEUI_save(t, n_inf_data, np.squeeze(yfit_inf[idx, :]), np.squeeze(yfit_a[idx, :]),
                       np.squeeze(yfit_e[idx, :]),
                       pop, result, r2_inf_tst, r2_inf_trn, adm, core_name + 'quarter' + str(adm), fixed_pars, idx, fit,
                       t_tst, dates,
                       mdl_u_i, mdl_u_r, useUnrepor)

        os.chdir(wd)


def plot_SEUI(t_trn, real_cuminf_trn, mdl_cuminf, mdl_asi, mdl_e, pop, result):
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
            plt.ylim([1, pop[0]])
            plt.yscale('log')
            plt.ylabel('Number of Cases')
    # fig1.suptitle('Population Compartments in '+adm+' - R2 = '+\
    #               '{:.3f}'.format(r2_inf))
    # fig1.savefig(adm+core_name+'_inf.png',dpi=250)

    plt.show()

    a = result.x[7]
    b = result.x[8]
    R_initU = result.x[3]
    R_redU = result.x[5] * R_initU
    RU = [sigmoid_R(R_initU, R_redU, a, b, t) for t in range(1, len(t_trn) + 1)]

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
    ax2.plot(RU, label='Reproductive number before symptoms')
    # ax2.set_xticks(ticks)
    # ax2.set_xticklabels(labels,rotation=45)
    plt.ylim([0, 3])
    plt.xlabel('Time [days]')
    plt.legend()

    return 0


def plot_SEUI_save(t_trn, real_cuminf_trn, mdl_cuminf, mdl_asi, mdl_e, pop, result, r2_inf_tst, r2_inf_trn, adm,
                   core_name, fixed_pars, idx,
                   fit, t_tst, dates, mdl_u_i, mdl_u_r, useUnrepor):
    n_adm = fixed_pars[0]
    n_cmp = fixed_pars[1]
    t_meas = fixed_pars[2]
    c = fixed_pars[4]
    local = fixed_pars[5]
    neigh = fixed_pars[7]
    alp_fix = fixed_pars[8]
    loadFr_dyn = fixed_pars[10]
    time_dep = fixed_pars[11]
    useSigTran = fixed_pars[12]
    T_inf = result[0].x[9 * n_adm + 1]
    alpha = result[0].x[9 * n_adm + 2]
    alp_soc = result[0].x[9 * n_adm + 3]
    if useUnrepor:
        T_inf_Ui = result[0].x[10 * n_adm + 4]
        p_unr = result[0].x[10 * n_adm + 5]

    # label sundays only
    ts = pd.to_datetime(dates)
    dates = ts.strftime('%m.%d').values
    # labels = [d for d in dates if d.weekday()==6]
    if len(t_tst > 0):
        ticks = np.arange(min(t_trn), max(t_tst), 7)
    else:
        ticks = np.arange(min(t_trn), max(t_trn), 7)
    fig1 = plt.figure(figsize=(12, 6))
    for i in range(2):
        ax = fig1.add_subplot(1, 2, i + 1)
        ax.plot(t_trn, real_cuminf_trn, 'gray', marker='o', markersize=5, \
                label='confirmed cases (training data)', zorder=10, linewidth=0)
        if len(t_tst) > 0:
            ax.plot(t_tst, real_cuminf_trn[-len(t_tst):], 'red', marker='o', markersize=5, \
                    label='confirmed cases (test data)', zorder=12, linewidth=0)
        ax.plot(t_trn, mdl_cuminf[:len(t_trn)], 'gray', linewidth=2, \
                label='confirmed infected+recovered cases (model)', zorder=15)
        ax.plot(t_trn, mdl_asi[:len(t_trn)], 'blue', linewidth=2, \
                label='asymptomatic infecteous (model)')
        ax.plot(t_trn, mdl_e[:len(t_trn)], 'green', linewidth=2, \
                label='exposed (model)')
        if useUnrepor:
            ax.plot(t_trn, mdl_u_i[:len(t_trn)], 'red', linewidth=2, \
                    label='unreported infecteous (model)')
            ax.plot(t_trn, mdl_u_r[:len(t_trn)], 'orange', linewidth=2, \
                    label='unreported recovered (model)')

        if len(result) > 1:
            T_inf_unc = []
            alpha_unc = []
            alp_soc_unc = []
            p_unr_unc = []
            T_inf_Ui_unc = []
            mean_ninf = np.zeros(mdl_cuminf.shape)
            for j in range(1, len(result)):
                mdl_sus_uncer = fit[j][0 * n_adm + idx]
                mdl_exp_uncer = fit[j][1 * n_adm + idx]
                mdl_unr_uncer = fit[j][2 * n_adm + idx]
                mdl_inf_uncer = fit[j][3 * n_adm + idx]
                mdl_cuminf_uncer = mdl_inf_uncer
                ax.plot(t_trn, mdl_cuminf_uncer[:len(t_trn)], '--', linewidth=0.5, color='grey', zorder=1)
                mean_ninf = mean_ninf + mdl_cuminf_uncer

                if useUnrepor:
                    T_inf_unc.append(result[j].x[9 * n_adm + 1])
                    alpha_unc.append(result[j].x[9 * n_adm + 2])
                    alp_soc_unc.append(result[j].x[9 * n_adm + 3])
                    T_inf_Ui_unc.append(result[j].x[10 * n_adm + 4])
                    p_unr_unc.append(result[j].x[10 * n_adm + 5])

                else:
                    T_inf_unc.append(result[j].x[7 * n_adm + 1])
                    alpha_unc.append(result[j].x[7 * n_adm + 2])
                    alp_soc_unc.append(result[j].x[7 * n_adm + 3])

            mean_ninf = mean_ninf / (len(result) - 1)
            T_inf_unc_mean = np.mean(T_inf_unc)
            alpha_unc_mean = np.mean(alpha_unc)
            alp_soc_unc_mean = np.mean(alp_soc_unc)
            T_inf_unc_std = np.std(T_inf_unc)
            alpha_unc_std = np.std(alpha_unc)
            alp_soc_unc_std = np.std(alp_soc_unc)

            if useUnrepor:
                T_inf_Ui_unc_mean = np.mean(T_inf_Ui_unc)
                T_inf_Ui_unc_std = np.std(T_inf_Ui_unc)
                p_unr_unc_mean = np.mean(p_unr_unc)
                p_unr_unc_std = np.std(p_unr_unc)

            ax.plot(t_trn, mean_ninf[:len(t_trn)], '-', linewidth=2, color='black', zorder=12,
                    label='Uncertainty average')

        ax.set_xticks(ticks)
        int_ticks = [int(i) for i in ticks]
        ax.set_xticklabels(dates[int_ticks], rotation=45)
        plt.xlabel('Time [days]')
        plt.grid()
        if i == 1:
            plt.ylim([0, 1.05 * max(max(real_cuminf_trn), max(real_cuminf_trn))])
            # plt.legend()
            plt.ylabel('Number of Cases')
        elif i == 0:
            plt.legend()
            plt.ylim([1, pop[0]])
            plt.yscale('log')
            plt.ylabel('Number of Cases')

            if len(result) > 1:
                plt.text(0.7, 90, 'alpha transport: {:.3f}'.format(alpha_unc_mean) + '+/- {:.3f}'.format(alpha_unc_std),
                         {'fontsize': 10})
                plt.text(0.7, 70,
                         'alpha social: {:.3f}'.format(alp_soc_unc_mean) + '+/- {:.3f}'.format(alp_soc_unc_std),
                         {'fontsize': 10})
                plt.text(0.7, 55, 'T_inf: {:.3f}'.format(T_inf_unc_mean) + '+/- {:.3f}'.format(T_inf_unc_std),
                         {'fontsize': 10})
                if useUnrepor:
                    plt.text(0.7, 43,
                             'T_infUR: {:.3f}'.format(T_inf_Ui_unc_mean) + '+/- {:.3f}'.format(T_inf_Ui_unc_mean),
                             {'fontsize': 10})
                    plt.text(0.7, 33, 'p_UR: {:.3f}'.format(p_unr_unc_mean) + '+/- {:.3f}'.format(p_unr_unc_mean),
                             {'fontsize': 10})

            else:
                plt.text(0.7, 90, 'alpha transport: {:.3f}'.format(alpha), {'fontsize': 10})
                plt.text(0.7, 70, 'alpha social: {:.3f}'.format(alp_soc), {'fontsize': 10})
                plt.text(0.7, 55, 'T_inf: {:.3f}'.format(T_inf), {'fontsize': 10})

                if useUnrepor:
                    plt.text(0.7, 43, 'T_infUR: {:.3f}'.format(T_inf_Ui), {'fontsize': 10})
                    plt.text(0.7, 33, 'p_UR: {:.3f}'.format(p_unr), {'fontsize': 10})

    # fig1.tight_layout()
    fig1.suptitle('Population Compartments in ' + str(adm) + ' - R2(test) = ' + \
                  '{:.3f}'.format(r2_inf_tst) + '- R2(train) = ' + \
                  '{:.3f}'.format(r2_inf_trn))
    fig1.savefig(str(adm) + core_name + '_inf.png', dpi=250)

    plt.show()

    if alp_fix:

        # if loadFr_dyn:
        #     R_initU = result[0].x[5 * n_adm:6 * n_adm]
        #     RU = [R_initU[idx] * time_dep(t) for t in t_trn]
        #     R_initU_i = result[0].x[9*n_adm+4:10*n_adm+4]
        #     RU_i = [R_initU_i[idx] * time_dep(t) for t in t_trn]
        # else:

        if useSigTran:

            if useUnrepor:

                a = result[0].x[8 * n_adm + 1:9 * n_adm + 1]
                b = result[0].x[7 * n_adm + 1:8 * n_adm + 1]
                Tinf = result[0].x[9 * n_adm + 1]
                TinfUi = result[0].x[10 * n_adm + 4]
                R_initU = result[0].x[5 * n_adm:6 * n_adm]
                R_initU_i = result[0].x[9 * n_adm + 4:10 * n_adm + 4]
                R_redU = result[0].x[6 * n_adm + 1:(6 * n_adm + 1) + n_adm]
            else:
                a = result[0].x[6 * n_adm + 1:7 * n_adm + 1]
                b = result[0].x[5 * n_adm + 1:6 * n_adm + 1]
                R_initU = result[0].x[5 * n_adm:6 * n_adm]

                R_redU = result[0].x[6 * n_adm + 1:(6 * n_adm + 1) + n_adm]

            RU = [sigmoid_R(R_initU[idx], R_redU[idx], a[idx], b[idx], t) for t in t_trn] / Tinf

            if useUnrepor:
                RU_i = [sigmoid_R(R_initU[idx], R_redU[idx], a[idx], b[idx], t) for t in t_trn] / TinfUi
                # [sigmoid_R(R_initU_i[idx], R_redU[idx], a[idx], b[idx], t) for t in t_trn]

            if loadFr_dyn:
                alpha = result[0].x[9 * n_adm + 2]
                alpha_use = [alpha * time_dep(t) for t in t_trn]


        else:

            a = result[0].x[6 * n_adm + 1:7 * n_adm + 1]
            b = result[0].x[5 * n_adm + 1:6 * n_adm + 1]
            R_initU = result[0].x[3 * n_adm:4 * n_adm]
            R_initU_i = result[0].x[9 * n_adm + 4:10 * n_adm + 4]
            R_redU = result[0].x[4 * n_adm + 1:5 * n_adm + 1]

            RU = [sigmoid_R(R_initU[idx], R_redU[idx], a[idx], b[idx], t) for t in t_trn]
            RU_i = [sigmoid_R(R_initU_i[idx], R_redU[idx], a[idx], b[idx], t) for t in t_trn]

        fig2 = plt.figure(figsize=(12, 6))

        ax2 = fig2.add_subplot(1, 2, 0 + 1)
        ax2.plot(t_trn, RU, label='Reproductive number before symptoms', zorder=15)

        all_a_uncert = [a]
        all_b_uncert = [b]
        all_R_initU_uncert = [R_initU]
        all_R_redU_uncert = [R_redU]
        allRU = np.zeros(np.array(RU).shape)
        allRUmax = max(RU)
        if len(result) > 1:

            for j in range(1, len(result)):
                Tinf_uncer = result[j].x[9 * n_adm + 1]
                TinfUi_uncer = result[j].x[10 * n_adm + 4]
                a_uncer = result[j].x[8 * n_adm + 1:9 * n_adm + 1]
                b_uncer = result[j].x[7 * n_adm + 1:8 * n_adm + 1]
                R_initU_uncer = result[j].x[5 * n_adm:6 * n_adm]
                R_redU_uncer = result[j].x[6 * n_adm + 1:7 * n_adm + 1]
                all_a_uncert.append(a_uncer)
                all_b_uncert.append(b_uncer)
                all_R_initU_uncert.append(R_initU_uncer)
                all_R_redU_uncert.append(R_redU_uncer)

                if useSigTran:
                    RU_uncer = [sigmoid_R(R_initU_uncer[idx], R_redU_uncer[idx], a_uncer[idx], b_uncer[idx], t) for t in
                                t_trn] / Tinf_uncer
                else:
                    RU_uncer = [sigmoid_R(R_initU_uncer[idx], R_redU_uncer[idx], a_uncer[idx], b_uncer[idx], t) for t in
                                t_trn]

                ax2.plot(t_trn, RU_uncer, '--', linewidth=0.5, color='grey', zorder=0.1 * j)
                allRU = allRU + RU_uncer
                allRUmax = max([allRUmax, max(RU_uncer)])

            # Mean of all runs
            allRU_mean = allRU / (len(result) - 1)
            ax2.plot(t_trn, allRU_mean, '-', linewidth=2, color='black', label='Mean reproductive number', zorder=10)

        ax2.set_xticks(ticks)
        int_ticks = [int(i) for i in ticks]
        ax2.set_xticklabels(dates[int_ticks], rotation=45)
        plt.xlabel('Time [days]')
        plt.ylabel('Reproductive Number')
        plt.ylim([0, 1.2 * allRUmax])
        plt.legend()

        if useUnrepor:
            ax1 = fig2.add_subplot(1, 2, 1 + 1)
            ax1.plot(t_trn, RU_i, label='Reproductive number unreported cases', zorder=15)

            all_R_initU_i_uncert = [R_initU]  # [R_initU_i]
            allRU_i = np.zeros(np.array(RU_i).shape)
            allRU_imax = RU_i.max()
            if len(result) > 1:

                for j in range(1, len(result)):
                    a_uncer = result[j].x[8 * n_adm + 1:9 * n_adm + 1]
                    b_uncer = result[j].x[7 * n_adm + 1:8 * n_adm + 1]
                    R_initU_i_uncer = result[j].x[5 * n_adm:6 * n_adm]  # result[j].x[9*n_adm+4:10*n_adm+44]
                    R_redU_uncer = result[j].x[6 * n_adm + 1:7 * n_adm + 1]
                    all_a_uncert.append(a_uncer)
                    all_b_uncert.append(b_uncer)
                    all_R_initU_i_uncert.append(R_initU_i_uncer)
                    all_R_redU_uncert.append(R_redU_uncer)

                    # if loadFr_dyn:
                    #     RU_i_uncer = [R_initU_i_uncer[idx] * time_dep(t) for t in t_trn]
                    # else:
                    if useSigTran:
                        RU_i_uncer = [sigmoid_R(R_initU_i_uncer[idx], R_redU_uncer[idx], a_uncer[idx], b_uncer[idx], t)
                                      for t in t_trn] / TinfUi_uncer
                    else:
                        RU_i_uncer = [sigmoid_R(R_initU_i_uncer[idx], R_redU_uncer[idx], a_uncer[idx], b_uncer[idx], t)
                                      for t in t_trn] / TinfUi_uncer

                    ax1.plot(t_trn, RU_i_uncer, '--', linewidth=0.5, color='grey', zorder=0.1 * j)
                    allRU_i = allRU_i + RU_i_uncer
                    allRU_imax = max([allRU_imax, max(RU_i_uncer)])

                # Mean of all runs
                allRU_i_mean = allRU_i / (len(result) - 1)
                ax1.plot(t_trn, allRU_i_mean, '-', linewidth=2, color='black', label='Mean reproductive number',
                         zorder=10)

            ax1.set_xticks(ticks)
            int_ticks = [int(i) for i in ticks]
            ax1.set_xticklabels(dates[int_ticks], rotation=45)
            plt.xlabel('Time [days]')
            plt.ylabel('Reproductive Number unreported cases')
            plt.ylim([0, 1.2 * allRU_imax])
            plt.legend()

        # fig2.tight_layout()
        fig2.suptitle('R in ' + str(adm))
        fig2.savefig(str(adm) + core_name + '_R.png', dpi=250)

        # Plot alpha
        fig3 = plt.figure(figsize=(6, 6))
        ax3 = fig3.add_subplot(1, 1, 1)
        ax3.plot(t_trn, alpha_use, label='Mobility rate', zorder=15)

        all_alpha = np.zeros(np.array(RU).shape)
        all_alpha_uncert = [alpha_use]
        if len(result) > 1:

            for j in range(1, len(result)):
                alpha_uncer = result[j].x[9 * n_adm + 2]
                all_alpha_uncert.append(alpha_uncer)
                alpha_uncer_use = [alpha_uncer * time_dep(t) for t in t_trn]

                ax3.plot(t_trn, alpha_uncer_use, '--', linewidth=0.5, color='grey', zorder=0.1 * j)
                all_alpha = all_alpha + alpha_uncer_use

            # Mean of all runs
            all_alpha = all_alpha / (len(result) - 1)
            ax3.plot(t_trn, all_alpha, '-', linewidth=2, color='black', label='Mean mobility rate', zorder=10)

        ax3.set_xticks(ticks)
        int_ticks = [int(i) for i in ticks]
        ax3.set_xticklabels(dates[int_ticks], rotation=45)
        plt.xlabel('Time [days]')
        plt.ylabel('Mobility rate')
        plt.ylim([0, max([3, 1.2 * all_alpha.max()])])
        plt.legend()
        fig3.suptitle('Mobility Rate')
        fig3.savefig(str(adm) + core_name + '_alpha.png', dpi=250)




    else:
        a = result[0].x[6 * n_adm + 1:7 * n_adm + 1]
        b = result[0].x[5 * n_adm + 1:6 * n_adm + 1]
        alpha_in = result[0].x[3 * n_adm:4 * n_adm]
        alpha_red = result[0].x[4 * n_adm + 1:5 * n_adm + 1]

        alpha = [sigmoid_R(alpha_in[idx], alpha_red[idx], a[idx], b[idx], t) for t in t_trn]

        fig2 = plt.figure(figsize=(12, 6))
        ax2 = fig2.add_subplot(121)
        ax2.plot(t_trn, alpha, label='Mobility rate alpha')

        all_a_uncert = [a]
        all_b_uncert = [b]
        all_R_initU_uncert = [alpha_in]
        all_R_redU_uncert = [alpha_red]
        if len(result) > 1:
            for j in range(1, len(result)):
                a_uncer = result[j].x[6 * n_adm + 1:7 * n_adm + 1]
                b_uncer = result[j].x[5 * n_adm + 1:6 * n_adm + 1]
                R_initU_uncer = result[j].x[3 * n_adm:4 * n_adm]
                R_redU_uncer = result[j].x[4 * n_adm + 1:5 * n_adm + 1]

                RU_uncer = [sigmoid_R(R_initU_uncer[idx], R_redU_uncer[idx], a_uncer[idx], b_uncer[idx], t) for t in
                            t_trn]
                ax2.plot(t_trn, RU_uncer, '--')

        # ax2.set_xticks(dates)
        # ax2.set_xticklabels(labels,rotation=45)
        plt.xlabel('Time [days]')
        plt.legend()
        fig2.suptitle('R in ' + str(adm))
        fig2.savefig(str(adm) + core_name + '_alpha.png', dpi=250)

    # df = pd.DataFrame(list(zip(all_a_uncert, all_b_uncert, all_R_initU_uncert,all_R_redU_uncert)),
    #                   columns=['a_R', 'b_R','R_initU','R_redU' ])
    # df.to_csv(str(adm) + core_name + '_uncert.csv')
    return 0


def save_data(adm0, adm1s, n_adm, core_name, fit, t_trn, t_tst, data_trn, data_tst, pop, start, end, result):
    '''Saves the data .csv files and figures per adm area. Prints R2.

    Parameters
    ----------
    adm0      : iso3 string of the country of interest
    adm1s     : vector with the analysed adm codes
    n_adm     : number of analyzed admin areas
    core_name : piece of the name of the csv file
    fit       : fit results matrix
    t_trn     : time points used for training
    t_tst     : time points reseved for testing
    data_trn  : data used for training
    data_tst  : data reseved for testing
    pop       : total population of the administrative area of interest
    start     : start date of analysis
    end       : end date of analysis
    '''
    # training data
    cuminf_trn = data_trn[1].reshape(n_adm, len(t_trn))
    fat_trn = data_trn[2].reshape(n_adm, len(t_trn))

    # test data
    cuminf_tst = data_tst[1].reshape(n_adm, len(t_tst))
    fat_tst = data_tst[2].reshape(n_adm, len(t_tst))

    # date labels for time axis
    dates = list(pd.date_range(start - dt.timedelta(1), end))

    if ('all' in adm1s):
        adm1s = np.array([adm0])

    for adm in adm1s:
        idx = np.where(np.array(adm1s) == adm)[0][0]

        mdl_sus = fit[0 * n_adm + idx]
        mdl_exp = fit[1 * n_adm + idx]
        mdl_inf = fit[2 * n_adm + idx]
        mdl_rec = fit[3 * n_adm + idx]
        mdl_fat = fit[4 * n_adm + idx]
        mdl_asi = fit[5 * n_adm + idx]
        mdl_asr = fit[6 * n_adm + idx]
        mdl_in2 = fit[7 * n_adm + idx]
        mdl_cuminf = mdl_inf + mdl_rec + mdl_fat + mdl_in2

        t = np.concatenate((t_trn, t_tst))

        os.chdir(os.path.join(wd, 'Results/fitting/'))
        save_csv(t, mdl_cuminf, adm, '_n_confirmed', core_name)
        save_csv(t, mdl_fat, adm, '_n_deceased', core_name)
        save_csv(t, mdl_rec, adm, '_n_recovered', core_name)
        save_csv(t, mdl_inf, adm, '_n_infected', core_name)
        save_csv(t, mdl_exp, adm, '_n_exposed', core_name)
        save_csv(t, mdl_sus, adm, '_n_susceptible', core_name)
        save_csv(t, mdl_asi, adm, '_n_asyminfected', core_name)
        save_csv(t, mdl_asr, adm, '_n_asymrecovered', core_name)
        save_csv(t, mdl_in2, adm, '_n_infected2', core_name)

        os.chdir(wd)

        real_cuminf_trn = cuminf_trn[idx]
        real_fat_trn = fat_trn[idx]
        real_cuminf_tst = cuminf_tst[idx]
        real_fat_tst = fat_tst[idx]

        os.chdir(os.path.join(wd, 'Results/original/'))
        save_csv(t_trn, real_cuminf_trn, adm, '_n_confirmed', core_name + '_trn')
        save_csv(t_trn, real_fat_trn, adm, '_n_deceased', core_name + '_trn')
        save_csv(t_tst, real_cuminf_tst, adm, '_n_confirmed', core_name + '_tst')
        save_csv(t_tst, real_fat_tst, adm, '_n_deceased', core_name + '_tst')
        os.chdir(wd)

        # R2
        r2_inf = r2_score(real_cuminf_trn, mdl_cuminf[:len(t_trn)])
        r2_fat = r2_score(real_fat_trn, mdl_fat[:len(t_trn)])
        print('\nR2 infected: %.3f, R2 deceased %.3f' % (r2_inf, r2_fat))

        real_lst = [real_cuminf_trn, real_fat_trn, real_cuminf_tst, real_fat_tst]
        mdl_lst = [mdl_cuminf, mdl_fat, mdl_sus, mdl_exp, mdl_inf, mdl_rec, mdl_asi, \
                   mdl_asr, mdl_in2]
        t_lst = [t_trn, t_tst]

        os.chdir(os.path.join(wd, 'Results/figures/'))
        plot_fit(adm, t_lst, real_lst, mdl_lst, core_name, r2_inf, r2_fat, pop, dates, \
                 result)
        os.chdir(wd)


def save_csv(t, y, adm, kind, core_name):
    '''
    Parameters
    ----------
    t         : time column to be stored
    y         : data column to be stored
    adm       : adm area code of the input data
    kind      : n_confirmed, n_deceased, etc.
    core_name : core file name
    '''
    df = pd.DataFrame(np.transpose([t, y]), columns=['timestamp', str(adm) + kind])
    df = df.astype({'timestamp': int})
    df.to_csv(str(adm) + kind + core_name + '.csv')


def save_pars(adm0, core_name, fitted, fixed):
    '''Saves model parameters in pickle format

    Parameters
    ----------
    adm0      : iso3 string of country of interest
    core_name : core name of file to be saved
    fitted    : optimized parameters
    fixed     : fixed parameters
    '''
    with open(adm0 + core_name + '_fitted.pkl', 'wb') as f:
        pickle.dump(fitted, f, pickle.HIGHEST_PROTOCOL)
    with open(adm0 + core_name + '_fixed.pkl', 'wb') as f:
        pickle.dump(fixed, f, pickle.HIGHEST_PROTOCOL)


# def sigmoid_R2(R_init, R_red, a, b, t):
#     #R_red = R_red_frac * R_init
#     R = (R_init - R_red) / (1 + np.exp(t - a) / b) + R_red
#     return R

def sigmoid_R3(R_init, R_red, a, b, t):
    # R_red = R_red_frac * R_init
    R = (R_init - R_red) / (1 + np.exp((t - a) / b)) + R_red / (1 + np.exp(- a / b))
    return R


def sigmoid_R(R_init, R_red_frac, a, b, t):
    # R_red = R_red_frac * R_init
    R = R_init * ((1. - R_red_frac) * (1 + np.exp(- a / b)) / (1 + np.exp((t - a) / b)) + R_red_frac)
    return R


def sigmoid_p(p_init, p_red_frac, a, b, t):
    # R_red = R_red_frac * R_init
    R = p_init * ((1. - p_red_frac) * (1 + np.exp(- a / b)) / (1 + np.exp((t - a) / b)) + p_red_frac)
    return R


def findFct(n, p):
    u_pr = n * np.tan(np.pi * p)
    u = u_pr / (1 + u_pr)

    return u


def timeDep_R(R_infU_in, R_redU_frac, t_dep):
    # R = R_infU_in * R_redU_frac ** (2 * (1 - t_dep))
    R = R_infU_in * (2 * t_dep * (1 - R_redU_frac) + 2 * R_redU_frac - 1)

    return R


def stretchFun(t, tmin, stretch):
    y = time_dep_soc(t) - t / tmin * stretch
    if type(t) == float:
        if y < 0:
            y = 0
    return y


def stretchFunPosNeg(t, tmin, stretch, time_dep_posneg):
    y = time_dep_posneg(t) - t / tmin * stretch
    # if y<0:
    #     y = 0
    return y


def stretchFunPosNeg2(t, tmin, stretch, time_dep_posneg):
    y = time_dep_posneg(t) - (t / tmin) * stretch
    # if y<0:
    #     y = 0
    return y


def sigmoid_R_withSlope(R_init, R_red_frac, a, b, ti):
    m = 0.001086
    R_red = R_red_frac * R_init
    R = (R_init - R_red) / (1 + np.exp(ti - a) / b) + R_red + m * ti

    return R


def plot_fit(adm, t_lst, real_lst, mdl_lst, core_name, r2_inf, r2_fat, pop, dates, result):
    '''Plots the fit and prediction on test data not.

    Parameters
    ----------
    t_lst     : list containing training and test time
    real_lst  : list containing observed data
    mdl_lst   : list containing modeled observables
    adm       : selected area to which plot results
    core_name : core name of the figure file
    r2_inf    : R2 in case of infected
    r2_fat    : R2 in case of deceased
    pop       : total population of the administrative area of interest
    dates     : date labels for time axis
    '''
    [real_cuminf_trn, real_fat_trn, real_cuminf_tst, real_fat_tst] = real_lst
    [mdl_cuminf, mdl_fat, mdl_sus, mdl_exp, mdl_inf, mdl_rec, mdl_asi, mdl_asr, \
     mdl_in2] = mdl_lst
    [t_trn, t_tst] = t_lst

    mdl_inf = mdl_inf + mdl_in2
    mdl_rec = mdl_rec + mdl_asr

    # label sundays only
    labels = [d for d in dates if d.weekday() == 6]
    ticks = np.linspace(7 - dates[0].weekday() - 2, \
                        len(labels) * 7 - dates[0].weekday() - 2, len(labels))
    labels = [d.strftime("%m-%d") for d in labels]

    fig1 = plt.figure(figsize=(12, 6))
    for i in range(2):
        ax = fig1.add_subplot(1, 2, i + 1)
        ax.plot(t_trn, real_cuminf_trn, 'gray', marker='o', markersize=3, \
                label='confirmed cases (data)')
        ax.plot(t_trn, real_fat_trn, 'red', marker='o', markersize=3, \
                label='deceased (data)')
        ax.plot(t_trn, mdl_cuminf[:len(t_trn)], 'gray', \
                label='confirmed cases (model)')
        ax.plot(t_trn, mdl_inf[:len(t_trn)], 'orange', \
                label='symptomatic infecteous (model)')
        ax.plot(t_trn, mdl_asi[:len(t_trn)], 'blue', \
                label='asymptomatic infecteous (model)')
        ax.plot(t_trn, mdl_rec[:len(t_trn)], 'green', \
                label='recovered (model)')
        ax.plot(t_trn, mdl_fat[:len(t_trn)], 'red', \
                label='deceased (model)')
        t = np.concatenate((np.array([t_trn[-1]]), t_tst))
        ax.plot(t_tst, mdl_cuminf[len(t_trn):], 'black')
        ax.plot(t_tst, mdl_asi[len(t_trn):], 'black')
        ax.plot(t_tst, mdl_inf[len(t_trn):], 'black')
        ax.plot(t_tst, mdl_rec[len(t_trn):], 'black')
    ax.plot(t_tst, mdl_fat[len(t_trn):], 'black')
    ax.plot(t_tst, real_cuminf_tst, 'black', marker='o', markersize=3)
    ax.plot(t_tst, real_fat_tst, 'black', marker='o', markersize=3)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45)
    plt.xlabel('Time [days]')
    plt.grid()
    if i == 1:
        plt.ylim([0, 1.05 * max(max(real_cuminf_trn), max(real_cuminf_tst))])
    elif i == 0:
        plt.legend()
        plt.ylim([1, pop[0]])
        plt.yscale('log')
        plt.ylabel('Number of Cases')

    fig1.suptitle('Population Compartments in ' + str(adm) + ' - R2 = ' + \
                  '{:.3f}'.format(r2_inf))
    fig1.savefig(str(adm) + core_name + '_inf.png', dpi=250)

    plt.close()

    R_init = result.x[-5]
    R_red = result.x[-8]
    a = result.x[-3]
    b = result.x[-4]
    R = [sigmoid_R(R_init, R_red, a, b, t) for t in range(1, len(dates) + 1)]

    fig2 = plt.figure(figsize=(12, 6))

    ax1 = fig2.add_subplot(121)
    ax1.plot(t_trn, real_fat_trn, 'red', marker='o', label='deceased (data)')
    ax1.plot(t_trn, mdl_fat[:len(t_trn)], 'r', label='deceased (model)')
    ax1.plot(t_tst, mdl_fat[len(t_trn):], 'black')
    ax1.plot(t_tst, real_fat_tst, 'ko')
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labels, rotation=45)
    plt.grid()
    plt.legend()
    plt.ylabel('Number of Dead')
    plt.xlabel('Time [days]')

    ax2 = fig2.add_subplot(122)
    ax2.plot(R, label='Asymptomatic reproductive number')
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labels, rotation=45)
    plt.xlabel('Time [days]')
    plt.legend()

    fig2.suptitle('Deaths in ' + str(adm) + '  -  R2 = ' + '{:.3f}'.format(r2_fat))
    fig2.savefig(str(adm) + core_name + '_fat.png', dpi=250)

    plt.close()


def plot_residuals(t, res_inf, res_dead, area, f_inf, f_dead):
    '''Plots the residuals (divided by their standard deviations) for both
    infected and fatalities for a single area

    Parameters
    ----------
    t        : vector of time points used for the fitting
    res_inf  : vector or residuals for infected
    res_dead : vector or residuals for n_fats
    area     : code of the area
    f_inf    : name of the file to be saved for the infected
    f_dead   : name of the file to be saved for the n_fats
    '''
    plt.figure()
    plt.plot(t, res_inf / np.std(res_inf))
    plt.ylabel('Residuals - Infected')
    plt.xlabel('Time [days]')
    plt.title('Residuals - Infections in ' + area)
    plt.savefig(f_inf)

    plt.figure()
    plt.plot(t, res_dead / np.std(res_dead))
    plt.ylabel('Residuals - Deaths')
    plt.xlabel('Time [days]')
    plt.title('Residuals - Deaths in ' + area)
    plt.savefig(f_dead)


################################################################################
# Other ILGE - not used anymore
################################################################################
def viralLoad_anaysis():
    filename = 'E:/Corona/Halbwertszeit_berechnen_visualisation_07.05.20 copy 4.xlsx'
    df = pd.read_excel(filename)
    df = df.iloc[0:21, :65]
    df_data = df.drop('days', axis=1)

    halflife = np.zeros((df_data.shape[1], 1))
    R2 = np.zeros((df_data.shape[1], 1))
    for i, col in enumerate(df_data.keys()):
        inds = df_data[col].notnull()
        v_loads = df_data[col].ix[inds]
        time = df.index.values[inds]
        max_ind = np.where(v_loads == max(v_loads))[0][0]
        loads_use = v_loads[max_ind:]
        time_use = time[max_ind:]

        # fit
        if len(time_use) > 2:
            [slope, offset] = np.polyfit(time_use, np.log(loads_use), 1)
            halflife[i] = np.log(0.5) / slope
            xplot = np.arange(0, 1.1 * max(time_use), 0.1)
            yplot = np.exp(offset) * np.exp(slope * xplot)
            R2[i] = r2_score(loads_use, np.exp(offset) * np.exp(slope * time_use))

            fig, ax = plt.subplots()
            plt.plot(time, v_loads, 'ro', label='All')
            plt.plot(time_use, loads_use, 'bo', label='Decay')
            plt.plot(xplot, yplot, 'k-', label='fit')
            ax.text(0.85, 0.95, 'Half life: %.2f' % halflife[i], horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)
            ax.text(0.85, 0.90, '$R^2$: %.2f' % R2[i], horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)

    # Plot histogram
    hl_use = halflife[halflife > 0]
    r2_use = R2[halflife > 0]
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(hl_use, bins=10)
    axs[0].title.set_text('Half life')
    axs[1].hist(r2_use, bins=10)
    axs[1].title.set_text('$R^2$')

    fig, ax = plt.subplots()
    plt.plot(hl_use, r2_use, 'ro', label='All')
    plt.xlabel('Half life')
    plt.ylabel('Coefficient of Determination')

    return 0


def loadFromCSV_ILGE():
    filename = 'E:/Corona/EpiData_ETH_censor_20200515.xlsx'
    df = pd.read_excel(filename, 'positive')
    df_neg = pd.read_excel(filename, 'negative')

    # Evaluate age group
    keys = df.keys()
    counts = df[keys[1]].value_counts().values
    labels = df[keys[1]].value_counts().keys().values
    counts_neg = df_neg[keys[1]].value_counts().values
    labels_neg = df_neg[keys[1]].value_counts().keys().values
    fig, a = plt.subplots(1, 2)
    a = a.ravel()
    for idx, ax in enumerate(a):
        if idx == 0:
            ax.bar(labels, counts)
            ax.text(0.85, 0.95, 'Total:' + str(np.sum(counts)), horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)
            ax.set_title('Positive')
        else:
            ax.bar(labels_neg, counts_neg)
            ax.text(0.85, 0.95, 'Total:' + str(np.sum(counts_neg)), horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)
            ax.set_title('Negative')
    plt.show()

    # Plot time series
    start = df['ENTNAHMEDATUM'].min()
    df['DELTA'] = df['ENTNAHMEDATUM'] - start
    counts_del = df['DELTA'].value_counts().values
    labels_del = df['DELTA'].value_counts().keys().values.astype('timedelta64[D]')
    sorted_counts = [x for _, x in sorted(zip(labels_del, counts_del))]
    sorted_labels = np.sort(labels_del)
    sorted_counts1, sorted_labels1 = getAGE_timeSeries(df, 1)
    sorted_counts2, sorted_labels2 = getAGE_timeSeries(df, 2)
    sorted_counts3, sorted_labels3 = getAGE_timeSeries(df, 3)
    sorted_counts4, sorted_labels4 = getAGE_timeSeries(df, 4)

    fig, ax = plt.subplots()
    plt.plot(sorted_labels, np.cumsum(sorted_counts), 'ro-', label='All')
    plt.plot(sorted_labels1, np.cumsum(sorted_counts1), 'bo-', label='0-7')
    plt.plot(sorted_labels2, np.cumsum(sorted_counts2), 'ko-', label='8-18')
    plt.plot(sorted_labels3, np.cumsum(sorted_counts3), 'go-', label='19-65')
    plt.plot(sorted_labels4, np.cumsum(sorted_counts4), 'yo-', label='>65')
    plt.xlabel('Time [days]')
    plt.ylabel('New Cases (cummulative)')
    plt.yscale('log')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    plt.plot(sorted_labels, sorted_counts, 'ro', label='All')
    plt.plot(sorted_labels1, sorted_counts1, 'bo', label='0-7')
    plt.plot(sorted_labels2, sorted_counts2, 'ko', label='8-18')
    plt.plot(sorted_labels3, sorted_counts3, 'go', label='19-65')
    plt.plot(sorted_labels4, sorted_counts4, 'yo', label='>65')
    plt.xlabel('Time [days]')
    plt.ylabel('New Cases')
    plt.legend()
    plt.show()

    return sorted_labels, np.cumsum(sorted_counts), sorted_labels1, np.cumsum(
        sorted_counts1), sorted_labels2, np.cumsum(sorted_counts2), \
           sorted_labels3, np.cumsum(sorted_counts3), sorted_labels4, np.cumsum(sorted_counts4)


def getAGE_timeSeries(df, age_cat):
    keys = df.keys()
    counts_del3 = df['DELTA'].loc[df[keys[1]] == age_cat].value_counts().values
    labels_del3 = df['DELTA'].loc[df[keys[1]] == age_cat].value_counts().keys().values.astype(
        'timedelta64[D]')
    sorted_counts3 = [x for _, x in sorted(zip(labels_del3, counts_del3))]
    sorted_labels3 = np.sort(labels_del3)

    return sorted_counts3, sorted_labels3


def load_csv(adm, kind, core_name):
    '''
    Parameters
    ----------
    t         : time column to be stored
    y         : data column to be stored
    adm       : adm area code of the input data
    kind      : n_confirmed, n_deceased, etc.
    core_name : core file name
    '''

    file = str(adm) + kind + core_name + '.csv'
    df = pd.read_csv(file)

    return df


def reload_data(run_ID):
    # general setup
    doplot = False
    mode = 'a'  # s: single, a: coupled
    start = dt.date(2020, 3, 7)  # start date of analysis (CHE:2/23,DEU:1/28)
    end = dt.date(2020, 4, 22)  # end date of analysis
    all_quat = list(np.arange(1, 21))  # the summarized quaters of Basel to be analysed: choose from 1-9,'all'
    # run_ID    = v#  Name for this specific run - folder will be created in Results

    dates = np.array(pd.date_range(start - dt.timedelta(1), end))
    ts = pd.to_datetime(dates)
    dates = ts.strftime('%m.%d').values
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

            # load data points

    # Get a,b
    a_s = np.empty((n_adm, len(result)))
    bs = np.empty((n_adm, len(result)))
    for k in range(0, len(result)):
        a = result[k].x[6 * n_adm + 1:7 * n_adm + 1]
        b = result[k].x[5 * n_adm + 1:6 * n_adm + 1]
        a_s[:, k] = a
        bs[:, k] = b

    if doplot:
        folder = os.path.join(wd, 'Results', run_ID, 'original')
        files = os.listdir(folder)
        folderfit = os.path.join(wd, 'Results', run_ID, 'fitting')
        filesfit = os.listdir(folderfit)
        for q in all_quat:
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

            # plot
            # label sundays only

            # labels = [d for d in dates if d.weekday()==6]
            t = list(t_trn) + list(t_tst)
            ticks = np.arange(min(t_trn), max(t_tst), 7)
            fig1 = plt.figure(figsize=(12, 6))
            for i in range(2):
                ax = fig1.add_subplot(1, 2, i + 1)
                ax.plot(t, n_inf, 'gray', linewidth=2, \
                        label='confirmed infected cases (model)', zorder=15)
                ax.plot(t, n_asy, 'blue', linewidth=2, \
                        label='asymptomatic infecteous (model)')
                ax.plot(t, n_exp, 'green', linewidth=2, \
                        label='exposed (model)')

                if len(result) > 1:
                    t = list(t_trn) + list(t_tst)
                    mean_ninf = np.zeros(n_inf.shape)
                    for j in range(1, len(result)):
                        fit = solution_SEUI(t, result[j].x, fixed_pars).T
                        mdl_sus_uncer = fit[0 * n_adm + q - 1]
                        mdl_exp_uncer = fit[1 * n_adm + q - 1]
                        mdl_unr_uncer = fit[2 * n_adm + q - 1]
                        mdl_inf_uncer = fit[3 * n_adm + q - 1]
                        mdl_cuminf_uncer = mdl_inf_uncer
                        ax.plot(t, mdl_cuminf_uncer, '--', linewidth=0.5, color='grey', zorder=1)
                        mean_ninf = mean_ninf + mdl_cuminf_uncer
                mean_ninf = mean_ninf / (len(result) - 1)
                ax.plot(t, mean_ninf, '-', linewidth=2, color='black', zorder=12)

                ax.set_xticks(ticks)
                int_ticks = [int(i) for i in ticks]
                ax.set_xticklabels(dates[int_ticks], rotation=45)
                plt.xlabel('Time [days]')
                plt.grid()
                if i == 1:
                    plt.ylim([0, 120])  # 1.05 * max(max(real_cuminf_trn), max(real_cuminf_trn))])
                elif i == 0:
                    plt.legend()
                    plt.ylim([1, 250000])
                    plt.yscale('log')
                    plt.ylabel('Number of Cases')
                ax.plot(t_trn, real_cuminf_trn, 'gray', marker='o', markersize=5, \
                        label='confirmed cases (training data)', zorder=10)
                ax.plot(t_tst, real_cuminf_tst, 'red', marker='o', markersize=5, \
                        label='confirmed cases (test data)', zorder=12)

            # fig1.tight_layout()
            r2_inf_trn = r2_score(real_cuminf_trn, n_inf[:len(t_trn)])
            print('\nR2 infected train: %.3f' % (r2_inf_trn))

            r2_inf_tst = r2_score(real_cuminf_tst, n_inf[len(t_trn):])
            print('\nR2 infected test: %.3f' % (r2_inf_tst))
            fig1.suptitle('Population Compartments in ' + str(q) + ' - R2(test) = ' + \
                          '{:.3f}'.format(r2_inf_tst) + '- R2(train) = ' + \
                          '{:.3f}'.format(r2_inf_trn))

            core_name = f[15:-4]
            os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
            fig1.savefig(str(q) + core_name + '_inf.png', dpi=250)
            os.chdir(wd)

            plt.show()

            if alpha_fix:
                idx = q - 1
                a = result[0].x[6 * n_adm + 1:7 * n_adm + 1]
                b = result[0].x[5 * n_adm + 1:6 * n_adm + 1]
                R_initU = result[0].x[3 * n_adm:4 * n_adm]
                R_redU = result[0].x[4 * n_adm + 1:5 * n_adm + 1]

                RU = [sigmoid_R(R_initU[idx], R_redU[idx], a[idx], b[idx], ti) for ti in t]

                fig2 = plt.figure(figsize=(12, 6))
                ax2 = fig2.add_subplot(121)
                ax2.plot(t, RU, label='Reproductive number before symptoms', zorder=15)

                all_a_uncert = [a]
                all_b_uncert = [b]
                all_R_initU_uncert = [R_initU]
                all_R_redU_uncert = [R_redU]
                allRU = np.zeros(np.array(RU).shape)
                all_R = np.zeros((len(RU), len(result)))
                all_R[:, 0] = RU
                if len(result) > 1:

                    for j in range(1, len(result)):
                        a_uncer = result[j].x[6 * n_adm + 1:7 * n_adm + 1]
                        b_uncer = result[j].x[5 * n_adm + 1:6 * n_adm + 1]
                        R_initU_uncer = result[j].x[3 * n_adm:4 * n_adm]
                        R_redU_uncer = result[j].x[4 * n_adm + 1:5 * n_adm + 1]
                        all_a_uncert.append(a_uncer)
                        all_b_uncert.append(b_uncer)
                        all_R_initU_uncert.append(R_initU_uncer)
                        all_R_redU_uncert.append(R_redU_uncer)

                        RU_uncer = [sigmoid_R(R_initU_uncer[idx], R_redU_uncer[idx], a_uncer[idx], b_uncer[idx], ti) for
                                    ti in
                                    t]
                        all_R[:, j] = RU_uncer
                        ax2.plot(t, RU_uncer, '--', linewidth=0.5, color='grey', zorder=0.1 * j)
                        allRU = allRU + RU_uncer

                    # Mean of all runs
                    allRU = allRU / (len(result) - 1)
                    ax2.plot(t, allRU, '-', linewidth=2, color='black', label='Mean reproductive number', zorder=10)

                ax2.set_xticks(ticks)
                int_ticks = [int(i) for i in ticks]
                ax2.set_xticklabels(dates[int_ticks], rotation=45)
                plt.xlabel('Time [days]')
                plt.ylabel('Reproductive Number')
                plt.legend()
                # fig2.tight_layout()
                fig2.suptitle('R in ' + str(q))
                os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
                fig2.savefig(str(q) + core_name + '_R.png', dpi=250)
                os.chdir(wd)
                os.chdir(os.path.join(wd, 'Results', run_ID, 'fitting'))
                df_RU = pd.DataFrame(data=all_R)
                df_RU.to_csv('_R_U' + core_name + 'quarter' + str(q) + '.csv')
                os.chdir(wd)

    return a_s, bs


def main():
    # Get social time series
    delta_t = 14

    df_timedep = pd.read_csv(os.path.join(wd, 'output', 'bs_full_traffic_model_timeseries.csv'))
    n_travelling = df_timedep['total'].values[:-2]

    # Starting 6.2. - relative to 7.3.
    time = 7 * df_timedep.index.values[:-2] - 30 + delta_t
    time_dep = UnivariateSpline(time, n_travelling)
    time_dep.set_smoothing_factor(0.0001)

    # Get social time series
    df_Kalman = pd.read_csv(os.path.join(wd, 'kalman', 'bs_kalman_Reff.csv'))
    time_Kalman = np.arange(0, 57)  # df_Kalman['timestamp'].values
    R_estimate = df_Kalman['R_estimate'].values

    if delta_t != 10:
        R_estimate = np.array(list(R_estimate[0] * np.ones(delta_t - 10, )) + list(R_estimate))
        time_Kalman = np.arange(0, len(R_estimate))

    global alpha_mob
    alpha_mob = time_dep(time_Kalman)
    y_Kalman = R_estimate / alpha_mob
    y_soc = y_Kalman / np.max(y_Kalman)

    global time_dep_soc
    time_dep_soc = UnivariateSpline(time_Kalman, y_soc, s=0.05)

    # Plot mobility
    fig1 = plt.figure(figsize=(6, 6))
    ax = fig1.add_subplot()
    plt.plot(time_Kalman, y_soc, '-', linewidth=2, label='social')
    plt.plot(time_Kalman, alpha_mob, '-', linewidth=2, label='mobility')
    plt.plot(time_Kalman, alpha_mob * y_soc, '-', linewidth=2, label='Product')
    plt.xlabel('Day since 26.2.2020')
    plt.ylabel('Interaction Fraction')
    plt.legend()

    # runIDs = ['Uncert_pfix0.92_Tinc2_2LivingSpace', 'Uncert_pfix0.92_Tinc2_2SENIOR_ANT',
    #           'Uncert_pfix0.92_Tinc2_21PHouseholds','Uncert_pfix0.92_Tinc2_2MedianIncome2017']
    runIDs = ['Final_uncert_20201204_newMedianIncome2017']
    # ['Final_uncert_20201204_LivingSpace',
    #       'Final_uncert_20201204_SENIOR_ANT',
    #       'Final_uncert_20201204_1PHouseholds',
    #       'Final_uncert_20201204_newMedianIncome2017']

    names = [
        'Median income']  # ['Living space\nper person','Seniority','1-person\nhouseholds','Median income'] # '1P Households', 'Seniority',
    # evalVaccination(runIDs, names)
    # plotMobilityBarChart(runIDs, names)
    for i, id in enumerate(runIDs):
        evalVaccination_new([id], [names[i]])
        # evalVaccination_ICUoccupation([id], [names[i]])
        # main_eval(id)

    return 0


def plotMobilityBarChart(runIDs, names):
    # general setup
    patterns = ["/", "+", ".", "x", "o", "O", "*", "\\", "|", "-", ]
    colors = ['orange', 'firebrick', 'gold', 'royalblue']
    t1 = []
    t2 = []
    t3 = []
    t1_inner = []
    t2_inner = []
    t3_inner = []
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
        t = list(np.arange(0, 120))

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

        Adj = fixed_pars[3].values
        t1.append(Adj[0, :].sum())
        t1_inner.append(Adj[0, 0].sum())
        t2.append(Adj[1, :].sum())
        t2_inner.append(Adj[1, 1].sum())
        t3.append(Adj[2, :].sum())
        t3_inner.append(Adj[2, 2].sum())

    # Plot
    x = np.arange(len(names))  # the label locations
    width = 0.15  # the width of the bars
    fig1 = plt.figure(figsize=(4, 4))
    ax = fig1.add_subplot(1, 1, 1)
    rects1 = ax.bar(x - 1.8 * width / 2, t1, width, label='T1', color=colors[1])
    rects2 = ax.bar(x, t2, width, label='T2', color=colors[2], hatch=patterns[0])
    rects3 = ax.bar(x + 1.8 * width / 2, t3, width, label='T3', color=colors[3], hatch=patterns[1])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Effective mobility contribution', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(prop={'size': 10}, frameon=False)
    plt.ylim([0, 0.6])
    plt.tight_layout()

    os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
    fig1.savefig('mobilitScoreBar_all.png', dpi=250)
    fig1.savefig('mobilitScoreBar_all.pdf', format='pdf')
    os.chdir(wd)

    return 0


def evalVaccination(runIDs, names):
    # general setup
    vaccineGroup = [3, 1, 3, 1]
    icu_stay = 5.9
    vaccinateALLsenior = False

    mode = 'a'  # s: single, a: coupled
    all_quat = [1, 2, 3]  # the summarized quaters of Basel to be analysed: choose from 1-9,'all'
    colors = ['black', 'darkblue', 'gold', 'royalblue', 'orange', 'yellow']
    fractions_vaccGroup = [0.3333333]
    effVacsGroup = [0.9]
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
                        ax.plot(t, mean_ninf, '-', linewidth=2, color=colors[counter], zorder=12,
                                label='V0 - no vaccine')

                        xV0_int = interp1d(mean_ninf, t)
                        xV0 = xV0_int(10000)

                        xV0_int = interp1d(mean_ninf, t)
                        yV0_int = interp1d(t, mean_ninf)

                        cases = []
                        for it in range(0, len(t) - int(round(icu_stay))):
                            cases.append((yV0_int(it + icu_stay) - yV0_int(it)) * 0.01)

                        xV0 = np.where(np.array(cases) > 22)[0][0] + icu_stay
                        # slope = np.gradient(mean_ninf)
                        # cutslope = 2200 / 14.5
                        # xV0_int = interp1d(slope, t)
                        # xV0 = xV0_int(cutslope)
                        yV0 = yV0_int(xV0)

                    else:
                        ax.plot(t, mean_ninf, '-', linewidth=2, color=colors[counter], zorder=12,
                                label='V1 - ' + f"{frac_vac * 100:.0f}" + '% randomly')  # with '+ f"{effVac*100:.0f}"+'% efficacy')
                        xV1_int = interp1d(mean_ninf, t)
                        yV1_int = interp1d(t, mean_ninf)
                        # slope = np.gradient(mean_ninf)
                        # cutslope = 2200 / 14.5
                        # xV1_int = interp1d(slope, t)
                        # xV1 = xV1_int(cutslope)
                        # yV1 = yV1_int(xV1)

                        cases = []
                        for it in range(0, len(t) - int(round(icu_stay))):
                            cases.append((yV1_int(it + icu_stay) - yV1_int(it)) * 0.01)

                        xV1 = np.where(np.array(cases) > 22)[0][0] + icu_stay
                        # slope = np.gradient(mean_ninf)
                        # cutslope = 2200 / 14.5
                        # xV0_int = interp1d(slope, t)
                        # xV0 = xV0_int(cutslope)
                        yV1 = yV1_int(xV1)

                    ax.fill_between(t, min_ninf, max_ninf, color=colors[counter], zorder=1, alpha=0.25, linewidth=0)

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
                if frac_vac == 0:
                    ax.plot(t, mean_ninf, '-', linewidth=2, color=colors[counter], zorder=12, label='V0 - no vaccine')
                else:
                    if i_fv == 1:
                        ax.plot(t, mean_ninf, '-', linewidth=2, color=colors[counter], zorder=12,
                                label='V2 - ' + f"{frac_vac * 100:.0f}" + '% T' + str(vaccineGroup[i_q]) + ' ' + names[
                                    i_q] + '=' + f"{fractions_vaccGroup[0] * 100:.0f}" + '% total')

                    else:
                        ax.plot(t, mean_ninf, '-', linewidth=2, color='red', zorder=12,
                                label='V2 - ' + f"{fractions_vaccGroup[0] * 100:.0f}" + '% from T' + str(
                                    vaccineGroup[i_q]) + "\n" + names[i_q])
                        # +f"{frac_vac * 100:.0f}" + '% T' + str(vaccineGroup[i_q]) +' '+  names[i_q] + '=' + f"{fractions_vaccGroup[0]*100:.0f}" + '% total')
                        # f"{frac_vac * 100:.0f}" + '% T' + str(
                        #     vaccineGroup[i_q]) + '=' + f"{fractions_vaccGroup[0]*100:.0f}" + '% total - '+names[i_q])

                xV2_int = interp1d(mean_ninf, t)
                # xV2 = xV2_int(10000)
                # xV2_int = interp1d(mean_ninf, t)
                # yV2_int = interp1d(t, mean_ninf)
                # slope = np.gradient(mean_ninf)
                # cutslope = 2200 / 14.5
                # xV2_int = interp1d(slope, t)
                # xV2 = xV2_int(cutslope)
                # yV2 = yV2_int(xV2)
                xV2_int = interp1d(mean_ninf, t)
                yV2_int = interp1d(t, mean_ninf)

                cases = []
                for it in range(0, len(t) - int(round(icu_stay))):
                    if vaccinateALLsenior:
                        cases.append((yV2_int(it + icu_stay) - yV2_int(it)) * 0.01 * 0.4)
                    else:
                        cases.append((yV2_int(it + icu_stay) - yV2_int(it)) * 0.01)

                xV2 = np.where(np.array(cases) > 22)[0][0] + icu_stay
                # slope = np.gradient(mean_ninf)
                # cutslope = 2200 / 14.5
                # xV0_int = interp1d(slope, t)
                # xV0 = xV0_int(cutslope)
                yV2 = yV2_int(xV2)

                ax.fill_between(t, min_ninf, max_ninf, color='red', zorder=3, alpha=0.25, linewidth=0)
                # plt.scatter(xV2, yV2, color='black')
                counter += 1

    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.xlabel('Time [days]', fontsize=14)
    plt.legend(prop={'size': 10}, frameon=True, facecolor='lightgrey')
    plt.ylim([0.1, 200000])
    plt.xlim([0, len(t) - 1])
    plt.yscale('log')
    plt.ylabel('Number of Cases', fontsize=14)
    ax.patch.set_facecolor('lightgrey')
    plt.grid(color='white')
    plt.scatter(xV0, yV0, color='black', marker='o', s=50, zorder=20)
    plt.scatter(xV1, yV1, color='darkblue', marker='o', s=50, zorder=20)
    plt.scatter(xV2, yV2, color='red', marker='o', s=50, zorder=20)
    plt.tight_layout()

    os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
    fig1.savefig('vaccineoneGroup_T3_mixed_all.png', dpi=250)
    fig1.savefig('vaccineoneGroup_T3_mixed_all.pdf', format='pdf')
    os.chdir(wd)

    print(xV0)
    print(xV1)
    print(xV2)
    print(yV0)
    print(yV1)
    print(yV2)

    return 0


def evalVaccination_ICUoccupation(runIDs, names):
    # general setup
    vaccineGroup = [3, 1, 3, 1]
    icu_stay = 5.9
    vaccinateALLsenior = True
    if vaccinateALLsenior:
        icu_perc2 = 0.005
    else:
        icu_perc2 = 1 * 0.01
    icu_perc = 1 * 0.01
    mode = 'a'  # s: single, a: coupled
    all_quat = [1, 2, 3]  # the summarized quaters of Basel to be analysed: choose from 1-9,'all'
    colors = ['black', 'darkblue', 'gold', 'royalblue', 'orange', 'yellow']
    fractions_vaccGroup = [0.3333333]
    effVacsGroup = [0.9]
    icu_capa = 44
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
                yV2 = 50  #
                xV2 = yV2_int_cases(50)

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

                cases_min = []
                cases_max = []
                yV0_int_min = interp1d(t, min_ninf)
                yV0_int_max = interp1d(t, max_ninf)
                cases = []
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
    # plt.hlines(y=50, xmin = 0, xmax =len(cases_min), linestyles = 'solid', color = 'darkorange')
    plt.scatter(xV0, yV0, color='black', marker='o', s=50, zorder=20)
    plt.scatter(xV1, yV1, color='darkblue', marker='o', s=50, zorder=20)
    plt.scatter(xV2, yV2, color='red', marker='o', s=50, zorder=20)
    plt.tight_layout()

    os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
    fig1.savefig('vaccineoneGroup_T' + str(vaccineGroup[i_q]) + '_ICUocc_mixed_all.png', dpi=250)
    fig1.savefig('vaccineoneGroup_T' + str(vaccineGroup[i_q]) + '_ICUocc_mixed_all.pdf', format='pdf')
    os.chdir(wd)

    print(xV0)
    print(xV1)
    print(xV2)
    print(yV0)
    print(yV1)
    print(yV2)

    return 0


def evalVaccination_new(runIDs, names):
    # general setup
    vaccineGroup = [1, 1, 3, 1]
    icu_stay = 5.9
    vaccinateALLsenior = False
    if vaccinateALLsenior:
        icu_perc2 = 0.005
    else:
        icu_perc2 = 1 * 0.01
    icu_perc = 1 * 0.01
    mode = 'a'  # s: single, a: coupled
    all_quat = [1, 2, 3]  # the summarized quaters of Basel to be analysed: choose from 1-9,'all'
    colors = ['black', 'darkblue', 'gold', 'royalblue', 'orange', 'yellow']
    fractions_vaccGroup = [0.3333333]
    effVacsGroup = [0.9]
    icu_capa = 44
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


def main_evalEffR(run_ID):
    # run_ID = 'multiplicModel_a8b5_LivingSpace'
    print(run_ID)
    plt.close('all')
    n_bootstraps = 50

    # general setup
    mode = 'a'  # s: single, a: coupled
    start = dt.date(2020, 2, 22)  # start date of analysis (CHE:2/23,DEU:1/28)
    end = dt.date(2020, 4, 22)  # end date of analysis
    all_quat = [1, 2, 3]  # the summarized quaters of Basel to be analysed: choose from 1-9,'all'

    colors = ['grey', 'red', 'green', 'blue']
    colorsMean = ['black', 'darkred', 'darkgreen', 'darkblue']
    fractions_vacc = [0, 0.25, 0.5, 0.75]

    global constantR
    global constantMobility
    global zeroMobility

    dates = np.array(pd.date_range(start - dt.timedelta(1), end))
    ts = pd.to_datetime(dates)
    dates = ts.strftime('%m.%d').values
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

    for q in all_quat:

        # Get data
        for i, f in enumerate(files):
            if f[0] == str(q) and f[-7:] == 'trn.csv':
                df_trn = pd.read_csv(os.path.join(folder, f))
                t_trn = df_trn['timestamp'].values

                if q == all_quat[0]:
                    real_cuminf_trn = df_trn[str(q) + '_n_confirmed'].values
                else:
                    real_cuminf_trn = real_cuminf_trn + df_trn[str(q) + '_n_confirmed'].values

    t = list(t_trn)
    ticks = np.arange(min(t_trn), max(t), 7)
    n_cmp = fixed_pars[1]

    # First plot fit results
    Tinf = result[0].x[(n_cmp - 1) * n_adm + 4 * n_adm + 1]
    TinfUi = result[0].x[(n_cmp - 1) * n_adm + 5 * n_adm + 4]
    T_inc = result[0].x[(n_cmp - 1) * n_adm + n_adm]

    maxN = max(real_cuminf_trn)

    fig1 = plt.figure(figsize=(6, 6))
    ax = fig1.add_subplot()
    for j in range(0, n_bootstraps):
        for q in range(0, len(all_quat)):

            fit = solution_SEUI(t, result[j].x, fixed_pars).T

            if q == 0:
                mdl_sus_uncer = fit[0 * n_adm + q]
                mdl_exp_uncer = fit[1 * n_adm + q]
                mdl_U_uncer = fit[2 * n_adm + q]
                mdl_Ui_uncer = fit[4 * n_adm + q]
                mdl_Ur_uncer = fit[5 * n_adm + q]
                mdl_cuminf_uncer = fit[3 * n_adm + q]
            else:
                mdl_sus_uncer += fit[0 * n_adm + q]
                mdl_exp_uncer += fit[1 * n_adm + q]
                mdl_U_uncer += fit[2 * n_adm + q]
                mdl_Ui_uncer += fit[4 * n_adm + q]
                mdl_Ur_uncer += fit[5 * n_adm + q]
                mdl_cuminf_uncer += fit[3 * n_adm + q]

        p_unr = mdl_Ur_uncer[-1] / (mdl_Ur_uncer[-1] + mdl_cuminf_uncer[-1])

        # ax.plot(t, mdl_exp_uncer,    '-', linewidth=2, color='green',  zorder=1, label='E')
        # ax.plot(t, mdl_U_uncer,      '-', linewidth=2, color='blue',zorder=1, label='P')
        # ax.plot(t, mdl_cuminf_uncer, '-', linewidth=2, color='grey', zorder=1,  label='I')
        # ax.plot(t, mdl_Ui_uncer,'-', linewidth=2, color='red', zorder=1, label='U_i')
        # ax.plot(t, mdl_Ur_uncer, '-', linewidth=2, color='yellow', zorder=1, label='U_r')
        # maxN = max([maxN,max(mdl_cuminf_uncer), max(mdl_U_uncer), max(mdl_exp_uncer)])
        #
        #
        #
        # #Format plot
        # ax.set_xticks(ticks)
        # plt.yticks(fontsize=16)
        # int_ticks = [int(i) for i in ticks]
        # ax.set_xticklabels(dates[int_ticks], rotation=45,fontsize=16)
        # plt.xlabel('Date',fontsize=14)
        # plt.grid()
        # ax.scatter(t_trn, real_cuminf_trn, color='black', marker='o', s=20, label='Data', zorder=10)
        # plt.legend(prop={'size': 16})
        # plt.ylim([0.001, 1.2*maxN])
        # plt.yscale('log')
        # plt.ylabel('Number of Cases',fontsize=16)
        # plt.tight_layout()
        # core_name = f[12:-4]
        # os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
        # fig1.savefig(str(q) + core_name + '_SUM_log.png', dpi=250)
        # os.chdir(wd)

        # Get effective reproductive numbe rfor each time step
        s = mdl_sus_uncer
        e = mdl_exp_uncer
        u = mdl_U_uncer
        ui = mdl_Ui_uncer
        ur = mdl_Ur_uncer
        i = mdl_cuminf_uncer
        n = s + e + u + ui + ur + i

        dedt = np.gradient(e)
        R_eff = (dedt + e / T_inc) / (u / Tinf + ui / TinfUi) * n / s

        if j == 0:
            ax.plot(t, R_eff, '-', linewidth=2, color='blue', zorder=1, label='Unditurbed Data')
        else:
            if j == 1:
                ax.plot(t, R_eff, '--', linewidth=0.5, color='grey', zorder=1, label='Bootstrap')
            else:
                ax.plot(t, R_eff, '--', linewidth=0.5, color='grey', zorder=1)

    ax.set_xticks(ticks)
    plt.yticks(fontsize=16)
    int_ticks = [int(i) for i in ticks]
    ax.set_xticklabels(dates[int_ticks], rotation=45, fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.grid()

    plt.legend(prop={'size': 16})
    plt.ylabel('Effective reproductive number', fontsize=16)
    plt.tight_layout()
    plt.ylim([0.1, 3])
    core_name = f[12:-4]
    os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
    fig1.savefig(str(q) + core_name + '_finalR_eff.png', dpi=250)
    os.chdir(wd)

    # Repeat separately for each tile
    for q in range(0, len(all_quat)):

        fit = solution_SEUI(t, result[0].x, fixed_pars).T
        mdl_sus_uncer = fit[0 * n_adm + q]
        mdl_exp_uncer = fit[1 * n_adm + q]
        mdl_U_uncer = fit[2 * n_adm + q]
        mdl_Ui_uncer = fit[4 * n_adm + q]
        mdl_Ur_uncer = fit[5 * n_adm + q]
        mdl_cuminf_uncer = fit[3 * n_adm + q]

        # Get effective reproductive numbe rfor each time step
        s = mdl_sus_uncer
        e = mdl_exp_uncer
        u = mdl_U_uncer
        ui = mdl_Ui_uncer
        ur = mdl_Ur_uncer
        i = mdl_cuminf_uncer
        n = s + e + u + ui + ur + i

        dedt = np.gradient(e)
        R_eff = (dedt + e / T_inc) / (u / Tinf + ui / TinfUi) * n / s

        # Get data
        for i, f in enumerate(files):
            if f[0] == str(q + 1) and f[-7:] == 'trn.csv':
                df_trn = pd.read_csv(os.path.join(folder, f))
                t_trn = df_trn['timestamp'].values

                real_cuminf_trn = df_trn[str(q + 1) + '_n_confirmed'].values

        # Only show R for times after the first case in this tile
        # R_eff[real_cuminf_trn<1] = np.nan

        fig1 = plt.figure(figsize=(6, 6))
        ax = fig1.add_subplot()
        ax.plot(t, R_eff, '-', linewidth=2, color='blue', zorder=1)
        ax.set_xticks(ticks)
        plt.yticks(fontsize=16)
        int_ticks = [int(i) for i in ticks]
        ax.set_xticklabels(dates[int_ticks], rotation=45, fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.grid()
        plt.ylim([0, 4])
        plt.ylabel('Effective reproductive number', fontsize=16)
        plt.tight_layout()
        core_name = f[12:-4]
        os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
        fig1.savefig(str(q) + core_name + '_finalR_eff.png', dpi=250)
        os.chdir(wd)

    # For each tile calculate the effective reproductive number in the absense of mobility/social interaction reduction

    global constantMobility
    global constantR
    constantR = True
    constantMobility = True
    fit = solution_SEUI(t, result[0].x, fixed_pars).T
    constantR = False
    constantMobility = False
    for q in range(0, len(all_quat)):
        mdl_sus_uncer = fit[0 * n_adm + q]
        mdl_exp_uncer = fit[1 * n_adm + q]
        mdl_U_uncer = fit[2 * n_adm + q]
        mdl_Ui_uncer = fit[4 * n_adm + q]
        mdl_Ur_uncer = fit[5 * n_adm + q]
        mdl_cuminf_uncer = fit[3 * n_adm + q]

        # Get effective reproductive numbe rfor each time step
        s = mdl_sus_uncer
        e = mdl_exp_uncer
        u = mdl_U_uncer
        ui = mdl_Ui_uncer
        ur = mdl_Ur_uncer
        i = mdl_cuminf_uncer
        n = s + e + u + ui + ur + i

        dedt = np.gradient(e)
        R_eff = (dedt + e / T_inc) / (u / Tinf + ui / TinfUi) * n / s

        # Get data
        for i, f in enumerate(files):
            if f[0] == str(q + 1) and f[-7:] == 'trn.csv':
                df_trn = pd.read_csv(os.path.join(folder, f))
                t_trn = df_trn['timestamp'].values

                real_cuminf_trn = df_trn[str(q + 1) + '_n_confirmed'].values

        fig2 = plt.figure(figsize=(6, 6))
        ax = fig1.add_subplot()
        ax.plot(t, mdl_exp_uncer, '-', linewidth=2, color='green', zorder=1, label='E')
        ax.plot(t, mdl_U_uncer, '-', linewidth=2, color='blue', zorder=1, label='P')
        ax.plot(t, mdl_cuminf_uncer, '-', linewidth=2, color='grey', zorder=1, label='I')
        ax.plot(t, mdl_Ui_uncer, '-', linewidth=2, color='red', zorder=1, label='U_i')
        ax.plot(t, mdl_Ur_uncer, '-', linewidth=2, color='yellow', zorder=1, label='U_r')
        maxN = max([maxN, max(mdl_cuminf_uncer), max(mdl_U_uncer), max(mdl_exp_uncer)])

        # Format plot
        ax.set_xticks(ticks)
        plt.yticks(fontsize=16)
        int_ticks = [int(i) for i in ticks]
        ax.set_xticklabels(dates[int_ticks], rotation=45, fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.grid()
        ax.scatter(t_trn, real_cuminf_trn, color='black', marker='o', s=20, label='Data', zorder=10)
        plt.legend(prop={'size': 16})
        plt.ylim([0.001, 1.2 * maxN])
        plt.yscale('log')
        plt.ylabel('Number of Cases', fontsize=16)
        plt.tight_layout()

        # Only show R for times after the first case in this tile
        R_eff[real_cuminf_trn < 1] = np.nan

        fig1 = plt.figure(figsize=(6, 6))
        ax = fig1.add_subplot()
        ax.plot(t, R_eff, '-', linewidth=2, color='blue', zorder=1)
        ax.set_xticks(ticks)
        plt.yticks(fontsize=16)
        int_ticks = [int(i) for i in ticks]
        ax.set_xticklabels(dates[int_ticks], rotation=45, fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.grid()
        plt.ylim([0, 4])
        plt.ylabel('Effective reproductive number', fontsize=16)
        plt.tight_layout()
        core_name = f[12:-4]
        os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
        fig1.savefig(str(q) + core_name + '_finalR_eff.png', dpi=250)
        os.chdir(wd)

        plt.close('all')

    return 0


def main_eval(run_ID):
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


def testmain():
    b = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5]
    a = 19
    Ri = 2.5
    Re = 0.5
    t_trn = np.linspace(0, 52)

    fig2 = plt.figure(figsize=(6, 6))
    ax2 = fig2.add_subplot(1, 1, 1)
    for i in range(0, len(b)):
        Rplot = [sigmoid_R(Ri, Re, a, b[i], t) for t in t_trn]
        ax2.plot(t_trn, Rplot, label=str(b[i]), zorder=15)
    plt.legend(prop={'size': 12}, loc="upper right")

    return 0


def main_eval_parallel(run_ID, tile, t, fit, params, fixed_pars, data_trn, t_trn):
    print(run_ID)

    # general setup
    saveParameters_tocsv = True
    plotfitR = True
    plotFitResult = True

    useSigTran = True
    useUnrepor = True

    start = dt.date(2020, 2, 26)  # start date of analysis (CHE:2/23,DEU:1/28)
    end = dt.date(2020, 4, 22)  # end date of analysis
    all_quat = [1, 2, 3]  # the summarized quaters of Basel to be analysed: choose from 1-9,'all'

    global constantR
    global constantMobility
    global zeroMobility

    dates = np.array(pd.date_range(start - dt.timedelta(1), end))
    ts = pd.to_datetime(dates)
    dates = ts.strftime('%m.%d').values
    n_adm = len(all_quat)

    # Save parameters
    for q in all_quat:
        # Save all parameters to csv
        a_save = []
        b_save = []
        Tinf_save = []
        TinfUi_save = []
        R_initU_save = []
        R_initU_i_save = []
        R_redU_save = []
        relR_redU_save = []
        relR_startU_save = []
        relR_redUi_save = []
        relR_startUi_save = []
        alpha_save = []
        a_save.append(params[8 * n_adm + 1 + q - 1])
        b_save.append(params[7 * n_adm + 1 + q - 1])
        Tinf_save.append(params[9 * n_adm + 1])
        TinfUi_save.append(params[10 * n_adm + 4])
        R_initU_save.append(params[5 * n_adm + q - 1])
        R_initU_i_save.append(params[9 * n_adm + 4 + q - 1])
        R_redU_save.append(params[6 * n_adm + 1 + q - 1] * params[5 * n_adm + q - 1])
        alpha_save.append(params[5 * n_adm + 4 * n_adm + 2])

        relR_redU_save.append(params[6 * n_adm + 1 + q - 1] * params[5 * n_adm + q - 1] / params[9 * n_adm + 1])
        relR_startU_save.append(params[5 * n_adm + q - 1] / params[9 * n_adm + 1])
        relR_redUi_save.append(params[6 * n_adm + 1 + q - 1] * params[5 * n_adm + q - 1] / params[10 * n_adm + 4])
        relR_startUi_save.append(params[5 * n_adm + q - 1] / params[10 * n_adm + 4])

        df = pd.DataFrame(
            list(zip(alpha_save, a_save, b_save, Tinf_save, TinfUi_save, R_initU_save, R_redU_save, relR_redU_save,
                     relR_startU_save, relR_redUi_save, relR_startUi_save)),
            columns=['alpha', 'a', 'b', 'Tinf', 'TinfUi', 'Rstart', 'Rend', 'relR_redU',
                     'relR_startU', 'relR_redUi', 'relR_startUi'])
        df.to_csv(os.path.join(wd, 'Results', run_ID, tile, 'parameters', str(q) + '_parameters.csv'))

    all_RU_i = []
    all_RUi_i = []
    all_RU_end = []
    all_RUi_end = []

    all_RU_i_abs = []
    all_RUi_i_abs = []
    all_RU_end_abs = []
    all_RUi_end_abs = []
    for q in all_quat:

        ticks = np.arange(min(t_trn), max(t), 7)

        # First plot fit results
        if plotFitResult:
            print('Plotting fit!')

            real_cuminf_trn = data_trn[q]  # 1:n_adm + 1, :

            mdl_exp_uncer = fit[1 * n_adm + q - 1]
            mdl_U_uncer = fit[2 * n_adm + q - 1]
            mdl_Ui_uncer = fit[4 * n_adm + q - 1]
            mdl_Ur_uncer = fit[5 * n_adm + q - 1]
            mdl_inf_uncer = fit[3 * n_adm + q - 1]
            mdl_cuminf_uncer = mdl_inf_uncer
            mdl_cumUinf_uncer = mdl_Ui_uncer + mdl_Ur_uncer

            thisr2 = r2_score(real_cuminf_trn, mdl_cuminf_uncer[:len(t_trn)])

            for i in range(2):
                fig1 = plt.figure(figsize=(6, 6))
                ax = fig1.add_subplot(1, 1, 1)

                maxN = max(mdl_cumUinf_uncer)
                r2_inf_trn = [r2_score(real_cuminf_trn, mdl_inf_uncer[:len(t_trn)])]
                print('\nR2 infected train: %.3f' % (r2_inf_trn[0]))
                ax.plot(t, mdl_exp_uncer, '-', linewidth=2, color='green', zorder=1, label='Exposed')
                ax.plot(t, mdl_U_uncer, '-', linewidth=2, color='blue', zorder=1, label='Presymptomatic')
                ax.plot(t, mdl_inf_uncer, '-', linewidth=2, color='red', zorder=1,
                        label='Infected+Recovered reported')
                ax.plot(t, mdl_cumUinf_uncer, '-', linewidth=2, color='grey', zorder=1,
                        label='Infected+Recovered unreported')
                ax.scatter(t_trn, real_cuminf_trn, color='black', marker='o', s=20, \
                           label='confirmed cases data', zorder=10)

                ax.set_xticks(ticks)
                plt.yticks(fontsize=16)
                int_ticks = [int(i) for i in ticks]
                ax.set_xticklabels(dates[int_ticks], rotation=45, fontsize=16, ha='right')
                # plt.xlabel('Time [days]',fontsize=14)
                plt.grid()
                if i == 1:

                    plt.ylim([0, 1.2 * maxN])  # 1.05 * max(max(real_cuminf_trn), max(real_cuminf_trn))])
                    plt.legend()
                    plt.ylabel('Number of Cases', fontsize=16)
                    plt.title('R^2 = ' + str(thisr2))
                    plt.tight_layout()
                    os.chdir(os.path.join(wd, 'Results', run_ID, tile, 'figures'))
                    fig1.savefig(str(q) + tile + '_inf_lin.png', dpi=250)
                    os.chdir(wd)


                elif i == 0:

                    plt.legend()
                    plt.ylim([1, 1.2 * maxN])
                    plt.yscale('log')
                    plt.ylabel('Number of Cases', fontsize=16)
                    plt.title('R^2 = ' + str(thisr2))
                    plt.tight_layout()
                    os.chdir(os.path.join(wd, 'Results', run_ID, tile, 'figures'))
                    fig1.savefig(str(q) + tile + '_inf_log.png', dpi=250)
                    os.chdir(wd)

        # Rel Reproductive number
        if plotfitR:
            print('Plotting R/Tinf!')
            idx = q - 1

            a = params[8 * n_adm + 1:9 * n_adm + 1]
            b = params[7 * n_adm + 1:8 * n_adm + 1]
            Tinf = params[9 * n_adm + 1]
            TinfUi = params[10 * n_adm + 4]
            R_initU = params[5 * n_adm:6 * n_adm]
            R_initU_i = params[9 * n_adm + 4:10 * n_adm + 4]
            R_redU = params[6 * n_adm + 1:(6 * n_adm + 1) + n_adm] * R_initU

            RU = [sigmoid_R(R_initU[idx], R_redU[idx], a[idx], b[idx], t) for t in t_trn] / Tinf
            RU_abs = [sigmoid_R(R_initU[idx], R_redU[idx], a[idx], b[idx], t) for t in t_trn]

            RU_i = [sigmoid_R(R_initU[idx], R_redU[idx], a[idx], b[idx], t) for t in t_trn] / TinfUi
            RU_i_abs = [sigmoid_R(R_initU[idx], R_redU[idx], a[idx], b[idx], t) for t in t_trn]

            fig2 = plt.figure(figsize=(12, 6))

            ax2 = fig2.add_subplot(1, 2, 0 + 1)
            ax2.plot(t_trn, RU, color='blue', label='Reproductive number before symptoms/Tinf', zorder=15)

            all_a_uncert = [a]
            all_b_uncert = [b]
            all_R_initU_uncert = [R_initU[idx] / Tinf]
            all_R_redU_uncert = [R_redU[idx] / Tinf]
            all_R_initU_uncert_abs = [R_initU[idx]]
            all_R_redU_uncert_abs = [R_redU[idx]]
            allRU = np.zeros(np.array(RU).shape)
            allRU_abs = np.zeros(np.array(RU).shape)
            allRUmax = max(RU)
            allRUmax_abs = max(RU_abs)

            ax2.set_xticks(ticks)
            int_ticks = [int(i) for i in ticks]
            ax2.set_xticklabels(dates[int_ticks], rotation=45, ha='right')
            # plt.xlabel('Time [days]',fontsize = 16)
            plt.ylabel('Reproductive Number/infection time', fontsize=16)
            plt.ylim([0, 1.2 * allRUmax])
            plt.legend(prop={'size': 12}, loc="upper right")
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)

            if useUnrepor:
                ax1 = fig2.add_subplot(1, 2, 1 + 1)
                ax1.plot(t_trn, RU_i, color='blue', label='Reproductive number unreported cases/TinfUi', zorder=15)

                all_R_initU_i_uncert = [R_initU[idx] / TinfUi]  # [R_initU_i]
                all_R_redU_i_uncert = [R_redU[idx] / TinfUi]
                allRU_i = np.zeros(np.array(RU_i).shape)
                allRU_imax = RU_i.max()
                all_R_initU_i_uncert_abs = [R_initU[idx]]  # [R_initU_i]
                all_R_redU_i_uncert_abs = [R_redU[idx]]
                allRU_i_abs = np.zeros(np.array(RU_i_abs).shape)
                allRU_imax_abs = np.array(RU_i_abs).max()

                ax1.set_xticks(ticks)
                int_ticks = [int(i) for i in ticks]
                ax1.set_xticklabels(dates[int_ticks], rotation=45, ha='right')
                # plt.xlabel('Time [days]',fontsize = 16)
                plt.ylabel('Reproductive Number/infection time', fontsize=16)
                plt.ylim([0, 1.2 * allRUmax])
                plt.legend(prop={'size': 12}, loc="upper right")
                plt.yticks(fontsize=16)
                plt.xticks(fontsize=16)
            plt.tight_layout()

            # fig2.tight_layout()
            os.chdir(os.path.join(wd, 'Results', run_ID, tile, 'figures'))
            fig2.savefig(str(q) + tile + '_relR.png', dpi=250)

            all_RU_i.append(all_R_initU_uncert)
            all_RUi_i.append(all_R_initU_i_uncert)
            all_RU_end.append(all_R_redU_uncert)
            all_RUi_end.append(all_R_redU_i_uncert)

            all_RU_i_abs.append(all_R_initU_uncert_abs)
            all_RUi_i_abs.append(all_R_initU_i_uncert_abs)
            all_RU_end_abs.append(all_R_redU_uncert_abs)
            all_RUi_end_abs.append(all_R_redU_i_uncert_abs)

        if plotfitR:
            print('Plotting R!')

            idx = q - 1
            a = params[8 * n_adm + 1:9 * n_adm + 1]
            b = params[7 * n_adm + 1:8 * n_adm + 1]
            Tinf = params[9 * n_adm + 1]
            TinfUi = params[10 * n_adm + 4]
            R_initU = params[5 * n_adm:6 * n_adm]
            R_initU_i = params[9 * n_adm + 4:10 * n_adm + 4]
            R_redU = params[6 * n_adm + 1:(6 * n_adm + 1) + n_adm] * R_initU

            RU = [sigmoid_R(R_initU[idx], R_redU[idx], a[idx], b[idx], t) for t in t_trn] / Tinf
            RU_abs = [sigmoid_R(R_initU[idx], R_redU[idx], a[idx], b[idx], t) for t in t_trn]
            RU_i = [sigmoid_R(R_initU[idx], R_redU[idx], a[idx], b[idx], t) for t in t_trn] / TinfUi
            RU_i_abs = [sigmoid_R(R_initU[idx], R_redU[idx], a[idx], b[idx], t) for t in t_trn]

            fig2 = plt.figure(figsize=(12, 6))

            ax2 = fig2.add_subplot(1, 2, 0 + 1)
            ax2.plot(t_trn, RU_abs, color='blue', label='Reproductive number before symptoms', zorder=15)

            all_a_uncert = [a]
            all_b_uncert = [b]
            all_R_initU_uncert = [R_initU[idx] / Tinf]
            all_R_redU_uncert = [R_redU[idx] / Tinf]
            all_R_initU_uncert_abs = [R_initU[idx]]
            all_R_redU_uncert_abs = [R_redU[idx]]
            allRU = np.zeros(np.array(RU).shape)
            allRU_abs = np.zeros(np.array(RU).shape)
            allRUmax = max(RU)
            allRUmax_abs = max(RU_abs)

            ax2.set_xticks(ticks)
            int_ticks = [int(i) for i in ticks]
            ax2.set_xticklabels(dates[int_ticks], rotation=45, ha='right')
            plt.xlabel('Time [days]', fontsize=16)
            plt.ylabel('Reproductive Number', fontsize=16)
            plt.ylim([0, 1.2 * allRUmax_abs])
            plt.legend(prop={'size': 12}, loc="upper right")
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)

            ax1 = fig2.add_subplot(1, 2, 1 + 1)
            ax1.plot(t_trn, RU_i_abs, color='blue', label='Reproductive number unreported cases', zorder=15)

            all_R_initU_i_uncert = [R_initU[idx] / TinfUi]  # [R_initU_i]
            all_R_redU_i_uncert = [R_redU[idx] / TinfUi]
            allRU_i = np.zeros(np.array(RU_i).shape)
            allRU_imax = RU_i.max()
            all_R_initU_i_uncert_abs = [R_initU[idx]]  # [R_initU_i]
            all_R_redU_i_uncert_abs = [R_redU[idx]]
            allRU_i_abs = np.zeros(np.array(RU_i_abs).shape)
            allRU_imax_abs = np.array(RU_i_abs).max()

            ax1.set_xticks(ticks)
            int_ticks = [int(i) for i in ticks]
            ax1.set_xticklabels(dates[int_ticks], rotation=45, ha='right')
            plt.xlabel('Time [days]', fontsize=16)
            plt.ylabel('Reproductive Number/infection time', fontsize=16)
            plt.ylim([0, 1.2 * allRUmax_abs])
            plt.legend(prop={'size': 12}, loc="upper right")
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.tight_layout()

            # fig2.tight_layout()
            os.chdir(os.path.join(wd, 'Results', run_ID, tile, 'figures'))
            fig2.savefig(str(q) + tile + '_R_abs.png', dpi=250)

            # os.chdir(wd)
            # os.chdir(os.path.join(wd, 'Results', run_ID, 'fitting'))
            # df_RU = pd.DataFrame(data=all_R)
            # df_RU.to_csv('_R_U' + core_name + 'quarter' + str(q) + '.csv')
            # os.chdir(wd)

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


def getBaselstrainCountOverTime():
    '''
    Parameters
    ----------
    the_quarter  : the areas of interest, may be 'all'
    start      : start date of analysis
    end        : end date of analysis

    Returns
    --------
    data matrix with time, n_fats, infected; and population for selected areas
    '''

    start = dt.date(2020, 2, 26)  # start date of analysis (CHE:2/23,DEU:1/28) - 3,7
    end = dt.date(2020, 4, 22)  # end date of analysis

    if not localMac:

        # read csv files for cases
        os.chdir('Data')
        filename = 'EpiData_ETH_complete_20201104_final.xlsx'
        df = pd.read_excel(filename, 'positive')
        os.chdir(wd)

        # Get time differences
        df['ENTNAHMEDATUM'] = df['ENTNAHMEDATUM'].dt.date
        START = df['ENTNAHMEDATUM'] >= start
        END = df['ENTNAHMEDATUM'] <= end
        df = df[START & END]
        tmp = (df['ENTNAHMEDATUM'] - start)
        df['DELTA'] = tmp.astype('timedelta64[D]')
        t = np.arange(df['DELTA'].max() + 1)  # np.array(np.unique(df['DELTA']))

        # Subset for the pangolin strain
        df_notBS = df[df['orf1b_C15324T'] == 'no']
        df_BS = df[df['orf1b_C15324T'] == 'yes']

        # Get dates
        ratios = []
        dates = df['DELTA'].unique()
        times = []
        n_B = []
        n_nB = []
        for d in dates:
            n_nB.append(len(df_notBS[df_notBS['DELTA'] == d]))
            n_B.append(len(df_BS[df_BS['DELTA'] == d]))

        # Create dataframe
        df_num = pd.DataFrame(list(zip(dates, n_B, n_nB)), columns=['time', 'n_B', 'n_nB'])
        df_num.to_csv(os.path.join(wd, 'plots', 'strains.csv'))

    else:
        df_num = pd.read_csv(os.path.join(wd, 'plots', 'strains.csv'))

    # Plot
    sum_n = df_num['n_B'].values + df_num['n_nB'].values
    inds = sum_n > 0
    ratio = df_num['n_B'].values[inds] / sum_n[inds]
    time = df_num['time'].values[inds]

    # 7 Day moving average
    y_inf_new_7av = np.zeros(y_inf.shape)
    for d in range(0, len(y_inf)):
        ind = np.arange(d - 3, d + 4)
        ind_use = ind[np.logical_and(ind >= 0, ind < len(y_inf))]
        y_inf_new_7av[d] = np.mean(np.array(y_new_inf)[ind_use])

    # save and plot
    plt.scatter(time, ratio)
    plt.xlabel('Time [days]')
    plt.ylabel('Fraction of orf1b_C15324T')
    plt.title('Fraction of orf1b_C15324T since 26.2.2020')
    plt.savefig(os.path.join(wd, 'plots', 'timeline.png'))

    df_numNew = pd.DataFrame(list(zip(list(time), list(ratio))), columns=['time', 'ratio'])
    df_numNew.to_csv(os.path.join(wd, 'plots', 'strains.csv'))

    return 0


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


def evalCityPopulation():
    df = pd.read_excel('/Users/sbrueningk/Downloads/worldcities.xlsx', 'Sheet2')
    df_g500k = df[df['population'] > 500000]
    df_g100k = df[df['population'] > 100000]
    df_g100ks500k = df_g100k[df_g100k['population'] <= 500000]
    return 0


################################################################################

if __name__ in "__main__":
    main()


# No longer in use!
def load_local_data_ILGE_MergeQuarters(the_quarter, start, end, usePangolin, pangolingStrain):
    '''
    Parameters
    ----------
    the_quarter  : the areas of interest, may be 'all'
    start      : start date of analysis
    end        : end date of analysis

    Returns
    --------
    data matrix with time, n_fats, infected; and population for selected areas
    '''

    # read csv files for cases
    os.chdir('/project/data/20200528T091303/content/epiData')
    # os.chdir('/Users/sbrueningk/Desktop/ILGE_Data/')
    filename = 'EpiData_ETH_censor_20200515.xlsx'
    df = pd.read_excel(filename, 'positive')
    df_neg = pd.read_excel(filename, 'negative')
    os.chdir(wd)

    # read soc'ioeconomic data
    os.chdir(os.path.join(wd, 'output'))
    # os.chdir('/Users/sbrueningk/Desktop/ILGE_Data/')
    # filename = 'SocioeconomicScore_data.xls'
    filename = 'bs_quarter_mapping.csv'
    pop_df = pd.read_csv(filename)
    os.chdir(wd)

    # restrict to dates greater than starting date
    df['ENTNAHMEDATUM'] = pd.to_datetime(df['ENTNAHMEDATUM'], format='%Y-%m-%d')
    df['ENTNAHMEDATUM'] = df['ENTNAHMEDATUM'].dt.date
    START = df['ENTNAHMEDATUM'] >= start
    END = df['ENTNAHMEDATUM'] <= end
    df = df[START & END]
    tmp = (df['ENTNAHMEDATUM'] - start)
    df['DELTA'] = tmp.astype('timedelta64[D]')
    t = np.array(np.unique(df['DELTA']))

    # Subset for selected adm1_pcode
    if the_quarter == 'all':
        pop = pop_df['POPULATION'].sum()
    else:

        # Get the quarters beloning to this ID
        quarters = pop_df['GEBIET'].loc[pop_df['ID_PARTNER'] == the_quarter].values
        print('Using these quarters:', quarters)

        # Get population
        pop = pop_df['POPULATION'].loc[pop_df['ID_PARTNER'] == the_quarter].sum()

        try:
            # Subset the case data
            df = df[df['Block ID'].isin(list(quarters))]
        except:
            print('USing all data since subset for quarter failed!!!')

    # Produce time series
    counts_del = df['DELTA'].value_counts().values
    labels_del = df['DELTA'].value_counts().keys().values.astype('timedelta64[D]')
    y_new_inf = [x for _, x in sorted(zip(labels_del, counts_del))]
    tsub = np.sort(labels_del)
    tsub = tsub.astype('timedelta64[D]') / np.timedelta64(1, 'D')
    for k in range(0, len(t)):
        if t[k] not in tsub:
            y_new_inf.insert(k, 0)

    y_inf = np.cumsum(y_new_inf)

    # Subset age group (not used at this stage yet)
    sorted_counts1, sorted_labels1 = getAGE_timeSeries(df, 1)
    sorted_counts2, sorted_labels2 = getAGE_timeSeries(df, 2)
    sorted_counts3, sorted_labels3 = getAGE_timeSeries(df, 3)
    sorted_counts4, sorted_labels4 = getAGE_timeSeries(df, 4)

    # So far no data for fatalities
    y_dead = []

    # Summarize
    data = []
    data.append(t)
    data.append(y_dead)
    data.append(y_inf)
    data.append(y_new_inf)

    return data, pop



