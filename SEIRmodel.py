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
wd = os.getcwd()
r2gof_threshRsq = 0.
useForTiles = 'MedianIncome2017'  # Choose from: 'LivingSpace', 'SENIOR_ANT','1PHouseholds','MedianIncome2017', 'random'
useForTilesList = ['LivingSpace', '1PHouseholds', 'SENIOR_ANT','MedianIncome2017']
useMultipeSeparations = False
randomIndex = '001'
n_jobs = 30
n_uncert = 50              # Number of runs to test uncertainty

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

global filenameCaseData
filenameCaseData = 'test.csv'


### MAIN #####_##################################################################

def main():

    # general setup
    start  = dt.date(2020, 2, 22)  # 26   # start date of analysis (CHE:2/23,DEU:1/28) - 3,7
    end    = dt.date(2020, 4, 22)  # end date of analysis
    n_tst  = 63                       # number of days reserved for testing
    uncert = True                   # Do a crude uncertainty estimation
    run_ID = 'Test_' + useForTiles  # Name for this specific run - folder will be created in Results
    useparallel = True              # number of parallel processes (for uncertainty analysis)
    delta_t = 14                    # time difference to the 7.3.2020 which is the date for which the time series data is normalized

    print('RUNNING: ' + run_ID)


    # Seed only a single case (exposed) - all other quarters start from 0!
    seedSingle = True
    seedQuarter = 1     # The quarter with the first case

    # Socioeconomic 'quarters'
    n_splitsSoc =               3  # number of splits for the socioeconomic data. Choose from 3 to 9, one section for 'NaN'
    all_quat    = list(np.arange(1, n_splitsSoc + 1))

    # initial values and constaints - separate depending on model chosen
    R_infU_in_0 = 7.88  # 2.  # transmission rate caused by symptomatic infecteous cases
    T_infU_0 = 3.  # length of infectious period in days fot those who recover
    T_inc_0 = 3.  # duration of incubation period in days (global)
    R_i_asy = .7  # Reproductive number for asymptomatic cases
    T_i_asy = 1.768
    p_asy = 0.88
    bnd_R_i_asy = ((0.6, 1.),)  # ((1., 5.),)
    bnd_T_i_asy = ((1.768, 1.768),)  # ((2., 12.),)
    bnd_p_asy = ((0.88, 0.88),)
    bnd_R_infU_in = ((0.1, 40.),)  # ((1., 5.),)
    bnd_T_infU = ((2.1, 2.1),)
    bnd_T_inc = ((2., 2.),)


    # Initial conditions
    n_exp_0 =  1   # initially exposed cases
    n_inf_0 = .0   # initially infecteous cases who will die or recover
    n_und_0 = .0   # initially infecteous who will recover
    n_un_i_0 = .0  # initially infecteous cases who will die or recover
    n_un_r_0 = .0  # initially infecteous who will recover
    bnd_n_exp = ((1, 1),)
    bnd_n_inf = ((0, 0),)
    bnd_n_und = ((0, 0),)
    bnd_n_uni = ((0, 0),)
    bnd_n_unr = ((0, 0),)



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


    # Actual model
    quat = all_quat

    if useMultipeSeparations:
        run_model_parallel(quat, newdir, t_meas, n_tst, start, end, uncert, n_uncert, run_ID, useparallel,
                           n_exp_0, n_inf_0, n_rec_0,n_fat_0, n_asi_0, n_asr_0, n_und_0,
                           R_infU_in_0, T_infI_0, T_inc_0, p_sym_0, T_infU_0,
                           bnd_n_exp, bnd_n_inf, bnd_n_rec, bnd_n_fat, bnd_n_asi, bnd_n_asr, bnd_n_und,
                           bnd_R_infU_in, bnd_T_infI, bnd_T_inc, bnd_p_sym, bnd_T_infU,
                           n_un_i_0, n_un_r_0, bnd_n_uni, bnd_n_unr,n_splitsSoc, seedSingle,seedQuarter,delta_t)
    else:
        run_model(quat, newdir, t_meas, n_tst, start, end, uncert, n_uncert, run_ID, useparallel,
                           n_exp_0, n_inf_0, n_rec_0,n_fat_0, n_asi_0, n_asr_0, n_und_0,
                           R_infU_in_0, T_infI_0, T_inc_0, p_sym_0, T_infU_0,
                           bnd_n_exp, bnd_n_inf, bnd_n_rec, bnd_n_fat, bnd_n_asi, bnd_n_asr, bnd_n_und,
                           bnd_R_infU_in, bnd_T_infI, bnd_T_inc, bnd_p_sym, bnd_T_infU,
                           n_un_i_0, n_un_r_0, bnd_n_uni, bnd_n_unr,n_splitsSoc, seedSingle,seedQuarter,delta_t)

        # Plot
        # try:
        main_eval(run_ID)
        # except:
        #     print('Failed at plotting!')

    return None


### FUNCTIONS #################################################################
def run_model_parallel(quat, newdir, t_meas, n_tst, start, end, uncert, n_uncert, run_ID, useparallel,
                           n_exp_0, n_inf_0, n_rec_0,n_fat_0, n_asi_0, n_asr_0, n_und_0,
                           R_infU_in_0, T_infI_0, T_inc_0, p_sym_0, T_infU_0,
                           bnd_n_exp, bnd_n_inf, bnd_n_rec, bnd_n_fat, bnd_n_asi, bnd_n_asr, bnd_n_und,
                           bnd_R_infU_in, bnd_T_infI, bnd_T_inc, bnd_p_sym, bnd_T_infU,
                           n_un_i_0, n_un_r_0, bnd_n_uni, bnd_n_unr,n_splitsSoc, seedSingle,seedQuarter,delta_t):
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


def run_model(quat, newdir, t_meas, n_tst, start, end, uncert, n_uncert, run_ID, useparallel,
                           n_exp_0, n_inf_0, n_rec_0,n_fat_0, n_asi_0, n_asr_0, n_und_0,
                           R_infU_in_0, T_infI_0, T_inc_0, p_sym_0, T_infU_0,
                           bnd_n_exp, bnd_n_inf, bnd_n_rec, bnd_n_fat, bnd_n_asi, bnd_n_asr, bnd_n_und,
                           bnd_R_infU_in, bnd_T_infI, bnd_T_inc, bnd_p_sym, bnd_T_infU,
                           n_un_i_0, n_un_r_0, bnd_n_uni, bnd_n_unr,n_splitsSoc, seedSingle,seedQuarter,delta_t):
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
    result  = []
    fit     = []
    data_in = [data_trn]
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
        result_i, t_i, fit_i = fit_general(par_list, data_trn, bnds, fixed_pars, adm0, t_trn, t_tst)
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
    save_data_SEUI(adm0, quat, n_adm, core_name, fit, t_trn, t_tst, data_trn, data_tst, \
                   pop, start, end, result, fixed_pars, newdir, useUnrepor)

    return 0


# Load data
def setup_ILGE(n_tst, start, end, quarter, n_splitsSoc, delta_t):
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

    n_cmp = 6       # Number of compartments


    # Get mobility time dependece
    # Load time searies - starts on 1.2.2020
    df_timedep = pd.read_csv(os.path.join(wd, 'output', 'bs_full_traffic_model_timeseries.csv'))
    n_travelling = df_timedep['total'].values[:-2]

    # Get the correct time frame and normalize by median number of travellers in Feb 2020
    # Starting 6.2. - relative to 7.3.
    time = 7 * df_timedep.index.values[:-2] - 30 + delta_t
    time_dep = UnivariateSpline(time, n_travelling)
    time_dep.set_smoothing_factor(0.0001)

    # Social interaction
    df_Kalman    = pd.read_csv(os.path.join(wd, 'kalman', 'bs_kalman_Reff.csv'))
    time_Kalman  = np.arange(0, 57)  # df_Kalman['timestamp'].values
    R_estimate   = df_Kalman['R_estimate'].values

    if delta_t != 10:
        R_estimate = np.array(list(R_estimate[0] * np.ones(delta_t - 10, )) + list(R_estimate))
        time_Kalman = np.arange(0, len(R_estimate))

    alpha_mob = time_dep(time_Kalman)
    y_Kalman = R_estimate / alpha_mob
    y_soc = y_Kalman / np.max(y_Kalman)
    global time_dep_soc
    time_dep_soc = UnivariateSpline(time_Kalman, y_soc, s=0.03)


    # parse the input data to the right format
    t_trn, t_tst, data_trn, data_tst, pop, data_trn_abs = \
        obtain_data_ILGE(quarter, n_tst, n_adm, start, end, usePangolin, pangolingStrain, mergeQuatSoc, n_splitsSoc)

    # obtain adjacency matrix and number of admin areas analyzed
    A = obtain_adjacencyILGE(quarter, neigh, n_splitsSoc)


    return n_cmp, A, n_adm, t_trn, t_tst, data_trn, data_tst, pop, ASoc, time_dep, data_trn_abs


def obtain_data_ILGE(quarter, n_tst, n_adm, start, end, n_splitsSoc):
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
    data_inf_tst = np.array([])
    data_inf_tst_cum = np.array([])
    pop = []

    for j, c in enumerate(quarter):
        print(c)

        data, subpop = load_data(c, start, end)

        if j == 0:
            if (n_tst > data[0][-1]):
                print('\nTest set larger than total dataset!\n')
            t_trn = data[0][:n_tst]
            t_tst = data[0][n_tst:]

        pop.append(subpop)

        # data (1st column is time, the rest are dead and infected, in couples)
        data_inf_trn_cum = np.concatenate((data_inf_trn_cum, data[2][:n_tst]))
        data_inf_tst_cum = np.concatenate((data_inf_tst_cum, data[2][n_tst:]))
        data_inf_trn = np.concatenate((data_inf_trn, data[3][:n_tst]))
        data_inf_tst = np.concatenate((data_inf_tst, data[3][n_tst:]))


    # training data: first row is time, then cummulative number of cases
    inf_trn = data_inf_trn.reshape(n_adm, len(t_trn))
    inf_trn_cum = data_inf_trn_cum.reshape(n_adm, len(t_trn))

    # test data
    inf_tst = data_inf_tst.reshape(n_adm, len(t_tst))


    # Impute missing data for the 29, 30, 31 of march
    inds_missing  = [36, 37, 38]
    inds_forRatio = [33, 34, 35, 19, 40, 41]
    ratioT = (inf_trn_cum / np.sum(inf_trn_cum, axis=0))[:, inds_forRatio].mean(axis=1)
    ind36 = np.where(t_poscases == 36)[0][0]
    cases = sequenced * baselstrain * posCases[ind36:ind36 + 3]
    inf_trn_imp = inf_trn.copy()
    for counter, i_miss in enumerate(inds_missing):
        for q_rep in range(0, n_adm):
            inf_trn_imp[q_rep, i_miss] = cases[counter] * ratioT[q_rep]

    # 7 day average
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

    data_trn = np.concatenate((t_trn[None, :], inf_trn_cum7av), axis=0)
    data_tst = np.concatenate((t_tst[None, :], inf_tst), axis=0)
    data_trn_abs = np.concatenate((t_trn[None, :], inf_trn_7av), axis=0)

    return t_trn, t_tst, data_trn, data_tst, pop, data_trn_abs


def load_data(the_quarter, start, end):
    '''
    Parameters
    ----------
    the_quarter  : the areas of interest, may be 'all'
    start        : start date of analysis
    end          : end date of analysis

    Returns
    --------
    data matrix with time, n_infected; and population for selected areas

    This function needs to be adjusted to hande your specific data set
    '''
    global randomIndex
    global ratio_time_dep



    # read csv files for cases
    os.chdir('Data')
    df = pd.read_excel(filenameCaseData, 'positive')
    os.chdir(wd)


    # read csv files for cases
    os.chdir('Data')
    df_pos = pd.read_excel(filename, 'positive')
    df_neg = pd.read_excel(filename, 'negative')
    os.chdir(wd)

    # restrict to dates greater than starting date
    df['ENTNAHMEDATUM'] = pd.to_datetime(df['ENTNAHMEDATUM'], format='%Y-%m-%d')
    df['ENTNAHMEDATUM'] = df['ENTNAHMEDATUM'].dt.date
    START = df['ENTNAHMEDATUM'] >= start
    END = df['ENTNAHMEDATUM'] <= end
    df = df[START & END]
    tmp = (df['ENTNAHMEDATUM'] - start)
    df['DELTA'] = tmp.astype('timedelta64[D]')
    t = np.arange(df['DELTA'].max())

    # Get all population data
    os.chdir(os.path.join(wd, 'geodata'))
    filename = 'SocioeconomicScore_data.csv'
    pop_df = pd.read_csv(filename)
    os.chdir(wd)

    # Get socioeconomic data
    name_suffix = 'percentiles'
    os.chdir(os.path.join(wd, 'graphs'))
    filename = 'bs_' + useForTiles + '_' + str(n_splitsSoc) + name_suffix + '.csv'
    soc_df = pd.read_csv(filename)
    os.chdir(wd)


    # Get total number of inhabitants
    os.chdir(os.path.join(wd, 'geodata'))
    filename = 'bs_quarter_mapping_all.csv'
    pop_tot_df = pd.read_csv(filename)
    os.chdir(wd)

    # For each tile get the blocks and population
    blocks = soc_df['BLO_ID'].loc[soc_df['percentile'] == the_quarter].values

    # Split according to living space
    if the_quarter == 0:
        otherBlocks = soc_df['BLO_ID'].loc[soc_df['percentile'] != the_quarter].values
        pop = pop_tot_df['POPULATION'].sum() - pop_df['Population 2017'].loc[
            pop_df['BlockID'].isin(otherBlocks)].sum()
    else:
        # Get population
        pop = pop_df['Population 2017'].loc[pop_df['BlockID'].isin(blocks)].sum()


    # Subset the case data
    df = df[df['Block ID'].isin(list(blocks))]
    print(str(the_quarter) + " has a total of " + str(df.shape[0]) + ' cases')

    # Generate epicurve
    counts_del = df['DELTA'].value_counts().values
    labels_del = df['DELTA'].value_counts().keys().values.astype('timedelta64[D]')
    y_new_inf = [x for _, x in sorted(zip(labels_del, counts_del))]
    tsub = np.sort(labels_del)
    tsub = tsub.astype('timedelta64[D]') / np.timedelta64(1, 'D')
    for k in range(0, len(t)):
        if t[k] not in tsub:
            y_new_inf.insert(k, 0)

    y_inf = np.cumsum(y_new_inf)


    # So far no data for fatalities
    y_dead = []

    # Summarize
    data = []
    data.append(t)
    data.append(y_dead)
    data.append(y_inf)
    data.append(y_new_inf)


    return data, pop, baselstrain


def obtain_adjacencyILGE(quarter, neigh, n_splitsSoc):
    global useForTiles

    transport_means = ['publ', 'bike', 'moto', 'foot']
    name_suffix     = 'percentiles'

    for i, tr in enumerate(transport_means):

        if useForTiles == 'MedianIncome2017':
            file = os.path.join(wd, 'graphs', 'bs_MedianIncome2017_' + str(
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

    # Normalize
    A = A_tr / A_tr.sum().sum()

    return A



# Parameters and bounds
def par_list_general(n_adm, n_exp_0, n_inf_0, n_und_0, R_infU_in_0, T_inc_0, T_infU_0, T_i_asy, p_asy,
                     n_un_i_0, n_un_r_0, seedSingle,ind_seedQuarter):


    # Mobility rate alpha is fixed but free, R is time-dependent
    par_list = initial_vars_SEUI(n_adm, n_exp_0, n_und_0, n_inf_0, n_un_i_0, n_un_r_0, seedSingle,ind_seedQuarter)

    par_list += list(np.repeat(R_infU_in_0, n_adm)) + [T_inc_0,T_infU_0,T_i_asy, p_asy]


    return par_list


def bnds_vars_general(n_adm, bnd_n_exp, bnd_n_inf, bnd_n_rec, bnd_n_fat, bnd_n_asi, bnd_n_asr, \
                      bnd_n_und, bnd_R_infU_in, bnd_T_inc, bnd_T_infU,
                      bnd_n_uni, bnd_n_unr, bnd_T_i_asy, bnd_p_asy, seedSingle, seedQuarter):

    bnds_lst = [bnd_n_exp, bnd_n_inf, bnd_n_und]
    bnds  = bnds_vars_SEUI(n_adm, bnds_lst, bnd_n_uni, bnd_n_unr, seedSingle, seedQuarter)

    bnds += bnd_R_infU_in * n_adm + bnd_T_inc + bnd_T_infU + bnd_T_i_asy + bnd_p_asy

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

        inits = [n_exp_0, n_und_0, n_inf_0, n_un_i_0, n_un_r_0]
        inits_all_adm = np.repeat(inits, n_adm)
        inits_all_adm = inits_all_adm.tolist()
        inits_all_adm[ind_seedQuarter] = 5.

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
    [bnd_n_exp, bnd_n_inf,bnd_n_in2] = bnds_lst

    if seedSingle:
        bnd_n_uni = ((0, 0),)
        bnd_n_unr = ((0, 0),)
        bnd_n_exp = ((0, 0),)
        bnd_n_inf = ((0, 0),)
        bnd_n_in2 = ((0, 0),)
        bnds = bnd_n_exp * n_adm + bnd_n_in2 * n_adm + bnd_n_inf * n_adm + bnd_n_uni * n_adm + bnd_n_unr * n_adm
        lst_bnds = list(bnds)
        lst_bnds[seedQuarter] = (0.1, 10)
        bnds = tuple(lst_bnds)
    else:
        bnds = bnd_n_exp * n_adm + bnd_n_in2 * n_adm + bnd_n_inf * n_adm + bnd_n_uni * n_adm + bnd_n_unr * n_adm

    return bnds


# Fit
def fit_general(par_list, data_trn, bnds, fixed_pars, adm0, t_trn, t_tst):

    result = dofit_SEUI(par_list, data_trn, bnds, fixed_pars, adm0)

    # optain curve resulting from optimization
    t = np.concatenate((t_trn, t_tst))
    fit = solution_SEUI(t, result.x, fixed_pars).T


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


def socFun(t):
    y = time_dep_soc(t)
    if type(t) == float:
        if y < 0:
            y = 0
    return y

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
    time_dep = fixed_pars[11]
    t_max = fixed_pars[16]

    # Optionally fit adjacency matrix
    Adj = fixed_pars[3]


    # Other fit parameters
    R_infU_in = pars[:n_adm]
    T_inc = pars[n_adm]
    T_infU = pars[4 * n_adm + 1]
    alpha  = 1
    T_inf_Ui = pars[5 * n_adm + 4]
    p_unr    = pars[5 * n_adm + 5]


    # Mobility
    if t < t_max:
        t_dep = time_dep(t)
    else:
        t_dep = time_dep(t_max)


    alpha_use = alpha * t_dep
    R_infU = socFun(t) * R_infU_in
    for i_tile in range(1, n_adm):
        R_infU[i_tile] = R_infU[0] * R_infU_in[i_tile]

    if constantMobility:
        alpha_use = alpha

        if zeroMobility:
            alpha_use = 0

    if constantR:
        R_infU[0] = R_infU_in[0].copy()
        for i_tile in range(1, n_adm):
            R_infU[i_tile] = R_infU_in[0] * R_infU_in[i_tile]

        R_infU = R_infU * fixedSocial

    # Same reproductive number for U and U_i
    R_inf_Ui = R_infU

    # compartments
    s = Y[0 * n_adm:1 * n_adm]
    e = Y[1 * n_adm:2 * n_adm]
    u = Y[2 * n_adm:3 * n_adm]
    i = Y[3 * n_adm:4 * n_adm]
    u_i = Y[4 * n_adm:5 * n_adm]
    u_r = Y[5 * n_adm:6 * n_adm]
    n   = s + e + i + u + u_i + u_r


    # Susceptibles: Add - diffusion term to E - diffusion term to A
    dsdt = - np.multiply(alpha_use * s * R_infU / (n), np.dot(Adj, np.multiply(1. / T_infU, u))) \
           - np.multiply(alpha_use * s * R_inf_Ui / (n), np.dot(Adj, np.multiply(1. / T_inf_Ui, u_i)))

    # Exposed - not infectious
    dedt = - np.multiply(1 / T_inc, e) \
           + np.multiply(alpha_use * s * R_infU / (n), np.dot(Adj, np.multiply(1. / T_infU, u))) \
           + np.multiply(alpha_use * s * R_inf_Ui / (n), np.dot(Adj, np.multiply(1. / T_inf_Ui, u_i)))

    # Presymptomatic
    dudt = np.multiply(1 / T_inc, e) - np.multiply(1 / T_infU, u)

    # Reported infected - assumed to be isolated
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
                 options={'gtol': 1e-8, 'ftol': 1e-8, 'disp': True, 'maxiter': 200})

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
    t = data[0]
    n_cuminf = data[1:n_adm + 1, :]

    # Solution with given parameter set
    yfit = solution_SEUI(t, pars, fixed_pars)
    yfit = yfit.T

    yfit_inf = yfit[3 * n_adm:4 * n_adm]
    yfit_cuminf = yfit_inf


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

    sol = odeint(lambda a, b, c: ode_model_SEUI_SymMat(a, b, c, fixed_pars), y0, t, args=(pars,))


    return sol


def uncertFit(input):
    data_trn    = input[0]
    t_trn       = input[1]
    par_list    = input[2]
    bnds        = input[3]
    fixed_pars  = input[4]
    adm0        = input[5]
    t_tst       = input[6]
    inds_s10    = input[7]
    rmse        = input[8]
    mean_points = input[9]
    rand_seed   = input[10]
    n_adm       = fixed_pars[0]

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
        result_i, t_i, fit_i = fit_general(par_list, data_new, bnds, fixed_pars, adm0, t_trn,t_tst)

        # Accept fit
        acceptableFit = 1


    out = [result_i, data_new, fit_i, acceptableFit, r2]
    return out






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








################################################################################

if __name__ in "__main__":
    main()

