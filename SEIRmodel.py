#!/usr/bin/env python3
#
# This is a simple compartmental epidemiology model (SEIR-model) describing
# the evolution of susceptible, exposed, presymptomatic, infected (Isolated) and unreported infected and recovered

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
useForTiles = 'MedianIncome2017'  # Choose from: 'LivingSpace', 'SENIOR_ANT','1PHouseholds','MedianIncome2017', 'random'
useForTilesList = ['LivingSpace', '1PHouseholds', 'SENIOR_ANT', 'MedianIncome2017']
useMultipeSeparations = False
n_jobs = 30
n_uncert = 50  # Number of runs to test uncertainty

global constantMobility
constantMobility = True

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


### MAIN #####_##################################################################

def main():
    # general setup
    start = dt.date(2020, 2, 22)  # 26   # start date of analysis (CHE:2/23,DEU:1/28) - 3,7
    end = dt.date(2020, 4, 22)  # end date of analysis
    uncert = False  # Do a crude uncertainty estimation
    run_ID = 'influenza_' + useForTiles  # Name for this specific run - folder will be created in Results
    useparallel = True  # number of parallel processes (for uncertainty analysis)

    # Seed only a single case (exposed) - all other quarters start from 0!
    seedSingle = True
    seedQuarter = 1  # The quarter with the first case

    # Socioeconomic 'quarters'
    n_splitsSoc = 3  # number of splits for the socioeconomic data. Choose from 3 to 9, one section for 'NaN'
    all_quat = list(np.arange(1, n_splitsSoc + 1))

    # initial values and constaints - separate depending on model chosen
    R_infU_in_0 = 7.88  # 2.  # transmission rate caused by symptomatic infecteous cases
    T_infU_0 = 3.  # length of infectious period in days fot those who recover
    T_inc_0 = 3.  # duration of incubation period in days (global)
    T_infI_0 = 3.
    p_sym_0 = .5
    bnd_R_infU_in = ((0.1, 40.),)
    bnd_T_infU = ((2.1, 2.1),)
    bnd_T_infI = ((2.1, 2.1),)
    bnd_T_inc = ((2., 2.),)
    bnd_p_sym = ((0., 1.),)

    # Initial conditions
    n_exp_0 = 1  # initially exposed cases
    n_inf_0 = .0  # initially infecteous cases who will die or recover
    n_und_0 = .0  # initially infecteous who will recover
    n_un_i_0 = .0  # initially infecteous cases who will die or recover
    n_un_r_0 = .0  # initially infecteous who will recover
    bnd_n_exp = ((1, 1),)
    bnd_n_inf = ((0, 0),)
    bnd_n_und = ((0, 0),)
    bnd_n_uni = ((0, 0),)
    bnd_n_unr = ((0, 0),)

    print('RUNNING: ' + run_ID)

    # Make output director
    newdir = wd + '/Influenza_Results/' + run_ID
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
    if useMultipeSeparations:

        # Optimization of all shared parameters
        run_model_parallel(all_quat, newdir, start, end, uncert, n_uncert, run_ID, useparallel,
                           n_exp_0, n_inf_0, n_und_0, n_un_i_0, n_un_r_0,
                           bnd_n_exp, bnd_n_inf, bnd_n_und, bnd_n_uni, bnd_n_unr,
                           R_infU_in_0, T_infI_0, T_inc_0, T_infU_0, p_sym_0,
                           bnd_R_infU_in, bnd_T_infI, bnd_T_inc, bnd_T_infU, bnd_p_sym,
                           n_splitsSoc, seedSingle, seedQuarter)
    else:

        # Run of a single partition
        run_model(all_quat, newdir, start, end, uncert, n_uncert, useparallel,
                  n_exp_0, n_inf_0, n_und_0, n_un_i_0, n_un_r_0,
                  bnd_n_exp, bnd_n_inf, bnd_n_und, bnd_n_uni, bnd_n_unr,
                  R_infU_in_0, T_infI_0, T_inc_0, T_infU_0, p_sym_0,
                  bnd_R_infU_in, bnd_T_inc, bnd_T_infU, bnd_T_infI, bnd_p_sym,
                  n_splitsSoc, seedSingle, seedQuarter)

    return None


### FUNCTIONS #################################################################
def run_model_parallel(quat, newdir, start, end, uncert, n_uncert, run_ID, useparallel,
                       n_exp_0, n_inf_0, n_und_0, n_un_i_0, n_un_r_0,
                       bnd_n_exp, bnd_n_inf, bnd_n_und, bnd_n_uni, bnd_n_unr,
                       R_infU_in_0, T_infI_0, T_inc_0, T_infU_0, p_sym_0,
                       bnd_R_infU_in, bnd_T_infI, bnd_T_inc, bnd_T_infU, bnd_p_sym,
                       n_splitsSoc, seedSingle, seedQuarter):
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
            setup(quat, start, end, n_splitsSoc)

        # assemble parameter list to be passed to optimizer
        ind_seedQuarter = quat.index(seedQuarter)
        par_list = par_list_general(n_adm, n_exp_0, n_inf_0, n_und_0, R_infU_in_0, T_inc_0, T_infU_0, T_i_asy, p_asy,
                                    n_un_i_0, n_un_r_0, seedSingle, ind_seedQuarter)

        # assemble optimization bounds
        bnds = bnds_vars_general(n_adm, bnd_n_exp, bnd_n_inf, bnd_n_und, bnd_R_infU_in, bnd_T_inc, bnd_T_infU,
                                 bnd_n_uni, bnd_n_unr, bnd_T_i_asy, bnd_p_asy, seedSingle, seedQuarter)

        # fixed paramters
        t_max = np.max(t_trn)
        fixed_pars = [n_adm, n_cmp, A, pop, time_dep, t_max]
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
    # Get fixed parameters
    fixed_pars_list = all_fixed_pars[0]
    fixed_pars_list.append(len(all_par_list))  # save number of separations
    n_adm = fixed_pars_list[0]
    n_cmp = fixed_pars_list[1]

    # Get parameters and reorganize
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
        R_infU_in = pars[n_adm * (n_cmp - 1):n_adm * (n_cmp - 1) + n_adm]
        startingCond = pars[:n_adm * (n_cmp - 1)]
        startCon_list = startCon_list + startingCond
        par_list = par_list + b_deR + a_deR + R_infU_in + R_redU_frac

        bnds_R_infU_in = list(bnds[n_adm * (n_cmp - 1):n_adm * (n_cmp - 1) + n_adm])
        bnds_list = bnds_list + bnds_b_deR + bnds_a_deR + bnds_R_infU_in + bnds_R_redU_frac
        startCon_bnds_list = startCon_bnds_list + list(bnds[:n_adm * (n_cmp - 1)])

        # Optionally fit adjacency matrix
        Adj = fixed_pars[3]
        fixed_pars_list.append(Adj)
        fixed_pars_list.append(fixed_pars[6])

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


# Fit only one
def run_model(quat, newdir, start, end, uncert, n_uncert, useparallel,
              n_exp_0, n_inf_0, n_und_0, n_un_i_0, n_un_r_0,
              bnd_n_exp, bnd_n_inf, bnd_n_und, bnd_n_uni, bnd_n_unr,
              R_infU_in_0, T_infI_0, T_inc_0, T_infU_0, p_sym_0,
              bnd_R_infU_in, bnd_T_inc, bnd_T_infU, bnd_T_infI, bnd_p_sym,
              n_splitsSoc, seedSingle, seedQuarter):
    # corresponding setup details
    n_cmp, A, n_adm, t_trn, t_tst, data_trn, data_tst, pop, ASoc, time_dep, data_trn_abs = setup(quat, start, end,
                                                                                                 n_splitsSoc)

    # assemble parameter list to be passed to optimizer
    ind_seedQuarter = quat.index(seedQuarter)
    par_list = par_list_general(n_adm, n_exp_0, n_inf_0, n_und_0, n_un_i_0, n_un_r_0,
                                R_infU_in_0, T_inc_0, T_infU_0, T_infI_0, p_sym_0,
                                seedSingle, ind_seedQuarter)

    # assemble optimization constraints
    bnds = bnds_vars_general(n_adm, bnd_n_exp, bnd_n_inf, bnd_n_und, bnd_n_uni, bnd_n_unr,
                             bnd_R_infU_in, bnd_T_inc, bnd_T_infU, bnd_T_infI, bnd_p_sym,
                             seedSingle, seedQuarter)

    # fixed paramters
    if len(t_tst) > 0:
        t_max = np.max([np.max(t_trn), np.max(t_tst)])
    else:
        t_max = np.max(t_trn)
    fixed_pars = [n_adm, n_cmp, A, pop, ASoc, time_dep, t_max]

    print(bnds)
    print(par_list)

    # Fit (optionally with uncertainty in data)
    result = []
    fit = []
    data_in = [data_trn]
    if uncert:

        # Fit (optionally with uncertainty in data)
        t_in = [t_trn]
        data_in = [data_trn]
        r2s = [0]

        # First run once with undisturbed data
        result_i, t_i, fit_i = fit_general(par_list, data_in[0], bnds, fixed_pars, t_in[0], t_tst)
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

        rmse = 0.3
        print('RMSE: ' + str(rmse))
        mean_points = data_trn_abs[1: n_adm + 1, :]

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
        input = [data_trn, t_trn, par_list_uncert, bnds_uncert, fixed_pars, t_tst, inds_s10, rmse, mean_points]
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

    else:
        # fit model
        result_i, t_i, fit_i = fit_general(par_list, data_trn, bnds, fixed_pars, t_trn, t_tst)
        result.append(result_i)
        fit.append(fit_i)
        print(result_i.x)

    # writing output
    core_name = '_quarter' + str(quat) + '_' + str(n_adm) + '_end-' + str(end) + 'uncert' + str(uncert)

    # save
    os.chdir(os.path.join(newdir, 'parameters'))
    save_pars(core_name, result, fixed_pars, bnds, data_in)
    os.chdir(wd)

    return 0


# Load data
def setup(quarter, start, end, n_adm):
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

    n_cmp = 6  # Number of compartments

    # Get mobility time dependece
    # Load mobility time series - starts on 1.2.2020
    df_timedep = pd.read_csv(filenameMobilityData)
    n_travelling = df_timedep['total'].values[:-2]

    # Get the time frame in days and normalize by median number of travellers
    time = 7 * df_timedep.index.values[:-2]
    time_dep = UnivariateSpline(time, n_travelling)
    time_dep.set_smoothing_factor(0.0001)

    # Social interaction
    df_Kalman = pd.read_csv(filenameKalmanData)
    time_Kalman = np.arange(0, 57)  # df_Kalman['timestamp'].values
    R_estimate = df_Kalman['R_estimate'].values

    # Get scores
    alpha_mob = time_dep(time_Kalman)
    y_Kalman = R_estimate / alpha_mob
    y_soc = y_Kalman / np.max(y_Kalman)
    global time_dep_soc
    time_dep_soc = UnivariateSpline(time_Kalman, y_soc, s=0.03)

    # parse the input data
    t_trn, data_trn, pop = obtain_data_ILGE(quarter, n_adm, start, end)

    # obtain adjacency matrix and number of admin areas analyzed
    A = obtain_adjacencyILGE(quarter)

    return n_cmp, A, n_adm, t_trn, t_tst, data_trn, data_tst, pop, ASoc, time_dep, data_trn_abs


def obtain_data_ILGE(quarter, n_adm, start, end):
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
    pop = []

    for j, c in enumerate(quarter):

        data, subpop = load_data(c, start, end)
        if j == 0:
            t_trn = data[0][:n_tst]

        # Population of this quarter
        pop.append(subpop)

        # Cases in this quarter: 1st column is time, the rest are cum. numbers of infected
        data_inf_trn_cum = np.concatenate((data_inf_trn_cum, data[2][:n_tst]))
        data_inf_trn = np.concatenate((data_inf_trn, data[3][:n_tst]))

    # Fit Data: first row is time, then cummulative number of cases
    inf_trn = data_inf_trn.reshape(n_adm, len(t_trn))
    inf_trn_cum = data_inf_trn_cum.reshape(n_adm, len(t_trn))

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

    # Final dataset
    data_trn = np.concatenate((t_trn[None, :], inf_trn_cum7av), axis=0)

    return t_trn, data_trn, pop


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


def obtain_adjacencyILGE(quarter):
    global useForTiles

    # Load mobility data for each transport mode, sum all
    transport_means = ['publ', 'bike', 'moto', 'foot']
    for i, tr in enumerate(transport_means):

        file = filenameSocioeconomicGraphData + '_' + tr + '.csv'
        A_tr = pd.read_csv(file)
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
def par_list_general(n_adm, n_exp_0, n_inf_0, n_und_0, n_un_i_0, n_un_r_0, R_infU_in_0, T_inc_0,
                     T_infU_0, T_infI_0, p_sym_0, seedSingle, ind_seedQuarter):
    # Mobility rate alpha is fixed but free, R is time-dependent
    par_list = initial_vars_SEUI(n_adm, n_exp_0, n_und_0, n_inf_0, n_un_i_0, n_un_r_0, seedSingle, ind_seedQuarter)

    par_list += list(np.repeat(R_infU_in_0, n_adm)) + [T_inc_0, T_infU_0, T_infI_0, p_sym_0]

    return par_list


def bnds_vars_general(n_adm, bnd_n_exp, bnd_n_inf, bnd_n_und, bnd_n_uni, bnd_n_unr,
                      bnd_R_infU_in, bnd_T_inc, bnd_T_infU, bnd_T_infI, bnd_p_sym,
                      seedSingle, seedQuarter):
    bnds_lst = [bnd_n_exp, bnd_n_inf, bnd_n_und]
    bnds = bnds_vars_SEUI(n_adm, bnds_lst, bnd_n_uni, bnd_n_unr, seedSingle, seedQuarter)

    bnds += bnd_R_infU_in * n_adm + bnd_T_inc + bnd_T_infU + bnd_T_infI + bnd_p_sym

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
    [bnd_n_exp, bnd_n_inf, bnd_n_in2] = bnds_lst

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
def fit_general(par_list, data_trn, bnds, fixed_pars, t_trn, t_tst):
    result = dofit_SEUI(par_list, data_trn, bnds, fixed_pars)

    # obtain curve resulting from optimization
    t = np.concatenate((t_trn, t_tst))
    fit = solution_SEUI(t, result.x, fixed_pars).T

    return result, t, fit


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
    alpha = 1
    T_inf_Ui = pars[5 * n_adm + 4]
    p_unr = pars[5 * n_adm + 5]

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
    n = s + e + i + u + u_i + u_r

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
    time_dep = fixed_pars[11]
    t_max = fixed_pars[16]

    n_separations = fixed_pars[17]
    n_parameters = fixed_pars[-1]

    # Parameters shared between separations
    alpha = 1
    T_infU = pars[4 * n_adm + 1]
    T_inf_Ui = pars[5 * n_adm + 4]
    T_inc = pars[n_adm]
    p_unr = pars[5 * n_adm + 5]

    for i_ind in range(0, n_separations):

        # Optionally fit adjacency matrix
        if i_ind == 0:

            # Other fit parameters specific to each separation
            R_infU_in = pars[:n_adm]
            Adj = fixed_pars[3]
        else:

            # Other fit parameters specific to each separation
            R_infU_in = pars[(i_ind - 1) * 4 * n_adm + n_parameters + 2 * n_adm:(
                                                                                            i_ind - 1) * 4 * n_adm + n_parameters + 3 * n_adm]
            Adj = fixed_pars[18 + (i_ind - 1) * 2]

        # Time dependence of mobility
        if t < t_max:
            t_dep = time_dep(t)
        else:
            t_dep = time_dep(t_max)
        alpha_use = alpha * t_dep

        # time dependence for measures taken on R
        R_infU = socFun(t) * R_infU_in

        # Relative reproductive number
        for i_tile in range(1, n_adm):
            R_infU[i_tile] = R_infU[0] * R_infU_in[i_tile]

        # Constant mobility
        if constantMobility:
            alpha_use = alpha
            if zeroMobility:
                alpha_use = 0

        # Constant social interaction
        if constantR:
            R_infU[0] = R_infU_in[0].copy()
            for i_tile in range(1, n_adm):
                R_infU[i_tile] = R_infU_in[0] * R_infU_in[i_tile]

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

        # Susceptibles
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


def dofit_SEUI(par_list, data_train, bnds, fixed_pars):
    result = \
        minimize(lambda var1, var2: residual_SEUI(var1, var2, fixed_pars), \
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


def residual_SEUI(pars, data, fixed_pars):
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
    data_trn = input[0]
    t_trn = input[1]
    par_list = input[2]
    bnds = input[3]
    fixed_pars = input[4]
    adm0 = input[5]
    t_tst = input[6]
    inds_s10 = input[7]
    rmse = input[8]
    mean_points = input[9]
    rand_seed = input[10]
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
        result_i, t_i, fit_i = fit_general(par_list, data_new, bnds, fixed_pars, adm0, t_trn, t_tst)

        # Accept fit
        acceptableFit = 1

    out = [result_i, data_new, fit_i, acceptableFit, r2]
    return out


# Save
def save_pars(core_name, fitted, fixed, bnds, data_in):
    '''Saves model parameters in pickle format

    Parameters
    ----------
    adm0      : iso3 string of country of interest
    core_name : core name of file to be saved
    fitted    : optimized parameters
    fixed     : fixed parameters
    '''

    with open(core_name + '_fitted.pkl', 'wb') as f:
        pickle.dump(fitted, f, pickle.HIGHEST_PROTOCOL)
    with open(core_name + '_fixed.pkl', 'wb') as f:
        pickle.dump(fixed, f, pickle.HIGHEST_PROTOCOL)
    with open(core_name + '_bounds.pkl', 'wb') as f:
        pickle.dump(bnds, f, pickle.HIGHEST_PROTOCOL)
    with open(core_name + '_datain.pkl', 'wb') as f:
        pickle.dump(data_in, f, pickle.HIGHEST_PROTOCOL)


################################################################################

if __name__ in "__main__":
    main()

