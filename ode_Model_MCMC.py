#!/usr/bin/env python3
# coding=utf-8
#
# This is a simple copmpartmental epidemiology model (SEIRD-model) describing
# the evolution of susceptible, exposed, infected, recovered and deceased
# population sizes.

### PACKAGES ###################################################################
import datetime          as dt
import matplotlib.pyplot as plt
import numpy             as np
import os
import pandas            as pd
import lmfit
import corner
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline



### GLOBAL VARIABLES ############################################################
wd = os.getcwd()
run_ID               = 'test'  # Name for this specific run - folder will be created in "Results
run_ID_loadFrom      = 'results_in_paper'  # Name for this specific run to load parameters from

# Default Options:
start                = dt.date(2020, 2, 22)  # start date of analysis
end                  = dt.date(2020, 4, 22)  # end date of analysis

p_unr_estimate       = 0.88 # Fraction of unreported cases - here estimated from serology data
useRandomTiles       = False # Option to use a random partition not based on socioeconomic or demographic data
useTinfUFix          = True # Option to include the infectious time for compartment u_i
doOpt                = True # Optimization using least squared with lmfit
doEMCEE              = True # Evaluation of parameter posterior distributions using MCMC
plotCases            = True # Plot the fit results (data and model simualtions for compartiments E,I,P and Ui+Ur)
plotR                = True # Plot temporal variation of the effective reproductive number
plot_mobility        = True # Very basic evaluation of the impact of mobility changes
plot_vaccine_individ = True # Evalaute vaccination scenarios for prioritization of some popualtion subgroups
plot_vaccine_all     = True # Evaluate vaccination scenarios when randomly vaccinating different population fractions
                            # and using differen vaccine efficacies

# Options for scenario simualtions - leave as is for now
global constantMobility
constantMobility = False

global zeroMobility
zeroMobility     = False

global constantR
constantR        = False

global fixedSocial
fixedSocial = 1

### DATA ############################################################

# Mobility time series
start_mobility     = dt.date(2020, 2, 6) # Date for whic mobility time series data starts
filename_mobility  = '' # include path to file

# Social interaction time series
start_kalman       = dt.date(2020, 2, 26)
kalmanFilterResult = '' # include path to file


# Mobility matrix files - one file for each mode of transport = ['publ', 'bike', 'moto', 'foot']
# File should contain absolute numbers of transport from A to B. These will then be added up and normalized.
# Naming convention: partitions_MobilityMatrix +mode of transport +'_mobility.csv' in a folder 'graphs' in the wd
# Example: 'MedianIncome_publ_mobility.csv'
mobilityMatrix_MedInc_prefix    = '' # include filename prefix only!
mobilityMatrix_seniority_prefix = '' # include filename prefix only!
mobilityMatrix_LivSpace_prefix  = '' # include filename prefix only!

# Socioeconomic data and population
# in a folder 'graphs' in the wd
filenameSocioeconomicData = '' # include path to file
filename_soc_MedInc       = '' # include path to file
filename_soc_Seniority    = '' # include path to file
filename_soc_LivSp        = '' # include path to file


# Case data
# data:         array containing the time vector (row 0), and 7-day moving window average absolute
#               case numbers for each tertile. Saved as .npy file.
# pop:          list containing the population per tile - here length 3!

# Random partition
randomIndex                = '001'
random_filename_data       = os.path.join(wd, 'Data', 'random'+randomIndex+'_data.npy')
mobilityMatrix_random_prefix ='' +randomIndex '' # include filename prefix only!
filename_soc_random        = '' # include path to file
pop_random                 = [n_0, n_1, n_2] # include population per tile

# Partitions based on socioeconomic data
data_MedInc_file          = '' # include path to file
pop_MedInc                =  [n_0, n_1, n_2] # include population per tile

data_seniority_file       = '' # include path to file
pop_Seniority             =  [n_0, n_1, n_2] # include population per tile

data_LivSp_file           = '' # include path to file
pop_LivSp                 =  [n_0, n_1, n_2] # include population per tile

### MAIN #####_##################################################################

def main():

    # general setup
    print('RUNNING: ' + run_ID)

    # Make output director
    newdir = wd + '/Results/' + run_ID
    if not os.path.exists(newdir):
        os.mkdir(newdir)
        os.chdir(newdir)
        os.mkdir('figures')
        os.mkdir('parameters')
        os.chdir(wd)


    # Evaluate model
    newfit()

    return 0

def newfit():

    # Get the time series data
    time_dep     = get_mobilityTimeSeries()
    time_dep_soc = get_SocialInteractionTimeSeries(time_dep)

    # Get Mobility matrix
    all_A = get_MobilityMatrix()
    n_adm = 3


    if useRandomTiles:

        # load data
        data       = np.load(random_filename_data)
        fixed_pars = [n_adm,time_dep,time_dep_soc,all_A[0],pop_random]

        # Define Parameters
        df_params = pd.read_csv(os.path.join(wd, 'Results', run_ID_loadFrom,
                                             'parameters', 'parameters_LSQfit.csv'))
        pars = lmfit.Parameters()

        # Constant parameters (shared)
        pars.add('E0', value=np.median(df_params['E0'].values), min=0.9999 * np.median(df_params['E0'].values),
                 max=np.median(df_params['E0'].values))
        pars.add('TinfUi', value=np.median(df_params['TinfUi'].values),
                 min=0.9999 * np.median(df_params['TinfUi'].values),
                 max=np.median(df_params['TinfUi'].values))
        pars.add('Tinc', value=np.median(df_params['Tinc'].values), min=0.9999 * np.median(df_params['Tinc'].values),
                 max=np.median(df_params['Tinc'].values))
        pars.add('TinfP', value=np.median(df_params['TinfP'].values), min=0.9999 * np.median(df_params['TinfP'].values),
                 max=np.median(df_params['TinfP'].values))

        # Parameters to be fit
        pars.add('R_T1', value=np.median(df_params['R_T1_Sen'].values), min=0.01, max=10.)
        pars.add('R_T2', value=np.median(df_params['R_T2_Sen'].values), min=0.01, max=2.)
        pars.add('R_T3', value=np.median(df_params['R_T3_Sen'].values), min=0.01, max=2.)


        # Do the fit
        out = lmfit.minimize(residual_SEUI_NEW, pars, args=(data, fixed_pars),
                             method='leastsq', nan_policy='omit')
        lmfit.printfuncs.report_fit(out.params, min_correl=0.5)


        # Resulting fir parameters
        this_pars = dict()
        if useTinfUFix:
            this_pars['TinfUi'] = 2.
        else:
            this_pars['TinfUi'] = out.params['TinfUi'].value
        this_pars['Tinc']  = out.params['Tinc'].value
        this_pars['TinfP'] = out.params['TinfP'].value
        this_pars['E0']    = out.params['E0'].value
        this_pars['R_T1']  = out.params['R_T1'].value
        this_pars['R_T2']  = out.params['R_T2'].value
        this_pars['R_T3']  = out.params['R_T3'].value
        this_pars['seed']  = 0

        # Get ODE solution for this parameter set
        n_adm = 3
        yfit = solution_SEUI_NEW(data[0], this_pars, fixed_pars).T
        yfit_inf = yfit[3 * n_adm:4 * n_adm]
        yfit_cuminf = yfit_inf
        colors = ['b', 'k', 'r']

        # Plot result
        plt.figure()
        plt.plot(data[0], data[1], 'ob')
        plt.plot(data[0], data[2], 'ok')
        plt.plot(data[0], data[3], 'or')
        for i_nad in range(0, 3):
            plt.plot(data[0], yfit_cuminf[i_nad, :], c=colors[i_nad])
        plt.legend(loc='best')
        plt.show()
        plt.savefig(os.path.join(wd, 'Results', run_ID, 'figures', 'cum_fit.png'))

        # Save parameters
        columns = out.params.keys()
        all_params = np.empty((1, len(columns)))
        for i_c, c in enumerate(columns):
            all_params[0, i_c] = out.params[c].value
        df_params_opt = pd.DataFrame(data=all_params, columns=columns)
        df_params_opt.to_csv(os.path.join(wd, 'Results', run_ID, 'parameters', 'parameters_fit.csv'))
    else:

        # Define parametes and load data:
        pars = lmfit.Parameters()

        # Median income
        data_MedInc       = np.load(data_MedInc_file)
        fixed_pars_MedInc = [n_adm,time_dep,time_dep_soc,all_A[0],pop_MedInc]
        pars.add('R_T1_MedInc', value=6, min=0.01, max=20.)
        pars.add('R_T2_MedInc', value=0.7, min=0.01, max=2.)
        pars.add('R_T3_MedInc', value=0.5, min=0.01, max=2.)


        # Living space per person
        data_Liv       = np.load(data_LivSp_file)
        fixed_pars_Liv = [n_adm,time_dep,time_dep_soc,all_A[1],pop_LivSp]
        pars.add('R_T1_Liv', value=6, min=0.01, max=20.)
        pars.add('R_T2_Liv', value=0.7, min=0.01, max=2.)
        pars.add('R_T3_Liv', value=0.5, min=0.01, max=2.)

        # Senior residents
        data_Sen       = np.load(data_seniority_file)
        fixed_pars_Sen = [n_adm,time_dep,time_dep_soc,all_A[2],pop_Seniority]
        pars.add('R_T1_Sen', value=6, min=0.01, max=20.)
        pars.add('R_T2_Sen', value=0.7, min=0.01, max=2.)
        pars.add('R_T3_Sen', value=0.5, min=0.01, max=2.)

        # Parameters shared for all partitions
        pars.add('E0', value=10, min=0.1, max=40.)
        if not useTinfUFix:
            pars.add('TinfUi', value=3., min=2., max=12.)
        pars.add('Tinc', value=3., min=2., max=7.)
        pars.add('TinfP', value=3., min=2., max=7.)

        # Combine data for parallel fit
        all_data       = [data_MedInc, data_Liv, data_Sen]
        all_fixed_pars = [fixed_pars_MedInc, fixed_pars_Liv, fixed_pars_Sen]
        i_datalst      = [0, 1, 2] # option to exclude some partitions
        names          = ['MedInc', 'LivSp', 'Sen']

        if doOpt:
            # Run optimization
            out = lmfit.minimize(residual_SEUI_NEW_parallel, pars, args=(all_data, all_fixed_pars),
                                 method='leastsq', nan_policy='omit')
            lmfit.printfuncs.report_fit(out.params, min_correl=0.5)
            for i_data in i_datalst:
                data = all_data[i_data]
                this_pars = dict()
                if useTinfUFix:
                    this_pars['TinfUi'] = 2.
                else:
                    this_pars['TinfUi'] = out.params['TinfUi']
                this_pars['Tinc']   = out.params['Tinc']
                this_pars['TinfP']  = out.params['TinfP']
                this_pars['E0']     = out.params['E0']
                this_pars['p_unr']  = p_unr_estimate

                if i_data == 0:
                    this_pars['seed'] = 0
                    this_pars['R_T1'] = out.params['R_T1_MedInc']
                    this_pars['R_T2'] = out.params['R_T2_MedInc']
                    this_pars['R_T3'] = out.params['R_T3_MedInc']
                elif i_data == 1:
                    this_pars['seed'] = 1
                    this_pars['R_T1'] = out.params['R_T1_Liv']
                    this_pars['R_T2'] = out.params['R_T2_Liv']
                    this_pars['R_T3'] = out.params['R_T3_Liv']
                elif i_data == 2:
                    this_pars['seed'] = 0
                    this_pars['R_T1'] = out.params['R_T1_Sen']
                    this_pars['R_T2'] = out.params['R_T2_Sen']
                    this_pars['R_T3'] = out.params['R_T3_Sen']

                # get solution
                n_adm = 3
                yfit = solution_SEUI_NEW(data[0], this_pars, all_fixed_pars[i_data]).T
                yfit_cuminf = yfit[3 * n_adm:4 * n_adm]


                # Plot result - just for brief check
                # Absolute data
                colors = ['b', 'k', 'r']
                plt.figure()
                plt.plot(data[0], np.gradient(data[1]), 'ob')
                plt.plot(data[0], np.gradient(data[2]), 'ok')
                plt.plot(data[0], np.gradient(data[3]), 'or')
                for i_nad in range(0, 3):
                    plt.plot(data[0], np.gradient(yfit_cuminf[i_nad, :]), c=colors[i_nad])
                plt.legend(loc='best')
                plt.show()

                # Cummulative data (compartment I)
                plt.figure()
                plt.plot(data[0], data[1], 'ob')
                plt.plot(data[0], data[2], 'ok')
                plt.plot(data[0], data[3], 'or')
                for i_nad in range(0, 3):
                    plt.plot(data[0], yfit_cuminf[i_nad, :], c=colors[i_nad])
                plt.legend(loc='best')
                plt.show()

                # Save fitted reproductive numbers and normalize by mobility contribution
                keys = ['R_T1', 'R_T2', 'R_T3']
                n_adm = 3
                R1_abs = []
                R2_abs = []
                R3_abs = []
                R1_eff = []
                R2_eff = []
                R3_eff = []
                dR1_abs = []
                dR2_abs = []
                dR3_abs = []
                dR1_eff = []
                dR2_eff = []
                dR3_eff = []
                for q in range(0, n_adm):

                    # Mobility contribution
                    norm = all_fixed_pars[i_data][3].values[q, :].sum()

                    if q > 0:
                        R  = this_pars[keys[q]].value * this_pars[keys[0]].value
                        dR = np.sqrt((this_pars[keys[q]].value * this_pars[keys[0]].stderr) ** 2 +
                                     (this_pars[keys[0]].value * this_pars[keys[q]].stderr) ** 2)
                    else:
                        R  = this_pars[keys[q]].value
                        dR = this_pars[keys[q]].stderr

                    if q == 0:
                        R1_abs.append(R)
                        R1_eff.append(R * norm)
                        dR1_abs.append(dR)
                        dR1_eff.append(dR * norm)
                    elif q == 1:
                        R2_abs.append(R)
                        R2_eff.append(R * norm)
                        dR2_abs.append(dR)
                        dR2_eff.append(dR * norm)
                    else:
                        R3_abs.append(R)
                        R3_eff.append(R * norm)
                        dR3_abs.append(dR)
                        dR3_eff.append(dR * norm)
                df_save = pd.DataFrame(list(zip(R1_abs, R2_abs, R3_abs, R1_eff, R2_eff, R3_eff,
                                                dR1_abs, dR2_abs, dR3_abs, dR1_eff, dR2_eff, dR3_eff)),
                                       columns=['R1_abs', 'R2_abs', 'R3_abs', 'R1_eff', 'R2_eff', 'R3_eff',
                                                'dR1_abs', 'dR2_abs', 'dR3_abs', 'dR1_eff', 'dR2_eff', 'dR3_eff'])
                df_save.to_csv(
                    os.path.join(wd, 'Results', run_ID, 'parameters', names[i_data] + '_R_fitResults.csv'))


            # Save all fitted parameters and fit uncertainties (from least squares)
            columns = list(out.params.keys())
            all_params = np.empty((1, len(columns)))
            all_dparams = np.empty((1, len(columns)))
            for i_c, c in enumerate(columns):
                all_params[0, i_c] = out.params[c].value
                all_dparams[0, i_c] = out.params[c].stderr
            df_params  = pd.DataFrame(data=all_params, columns=columns)
            df_dparams = pd.DataFrame(data=all_params, columns=columns)
            df_params.to_csv(os.path.join(wd, 'Results', run_ID, 'parameters', 'parameters_LSQfit.csv'))
            df_dparams.to_csv(os.path.join(wd, 'Results', run_ID, 'parameters', 'dparameters_LSQfit.csv'))

            # Add lnsigma for emceee
            out.params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2))
            params_inemcee = out.params
        else:

            # Load previously saved fit results
            df_fit = pd.read_csv(os.path.join(wd, 'Results', run_ID_loadFrom,
                                              'parameters', 'parameters_LSQfit.csv'))
            params_inemcee = lmfit.Parameters()
            params_inemcee.add('R_T1_MedInc', value=df_fit['R_T1_MedInc'].values[0], min=0.01, max=20.)
            params_inemcee.add('R_T2_MedInc', value=df_fit['R_T2_MedInc'].values[0], min=0.01, max=2.)
            params_inemcee.add('R_T3_MedInc', value=df_fit['R_T3_MedInc'].values[0], min=0.01, max=2.)
            params_inemcee.add('R_T1_Liv', value=df_fit['R_T1_Liv'].values[0], min=0.01, max=20.)
            params_inemcee.add('R_T2_Liv', value=df_fit['R_T2_Liv'].values[0], min=0.01, max=2.)
            params_inemcee.add('R_T3_Liv', value=df_fit['R_T3_Liv'].values[0], min=0.01, max=2.)
            params_inemcee.add('R_T1_Sen', value=df_fit['R_T1_Sen'].values[0], min=0.01, max=20.)
            params_inemcee.add('R_T2_Sen', value=df_fit['R_T2_Sen'].values[0], min=0.01, max=2.)
            params_inemcee.add('R_T3_Sen', value=df_fit['R_T3_Sen'].values[0], min=0.01, max=2.)

            if not useTinfUFix:
                params_inemcee.add('TinfUi', value=df_fit['TinfUi'].values[0], min=2., max=12.)
            params_inemcee.add('E0', value=df_fit['E0'].values[0], min=0.1, max=40.)
            params_inemcee.add('Tinc', value=df_fit['Tinc'].values[0], min=2., max=12.)
            params_inemcee.add('TinfP', value=df_fit['TinfP'].values[0], min=2., max=12.)
            params_inemcee.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2))

        if doEMCEE:

            # MCMC to evaluate parameter posterior distribution - use at least 100-1000 steps
            res = lmfit.minimize(residual_SEUI_NEW_parallel, args=(all_data, all_fixed_pars),
                                 method='emcee', nan_policy='omit', steps=100,
                                 params=params_inemcee, is_weighted=False, progress=True,
                                 workers=20)


            # Save the parameter combinations
            columns = list(res.flatchain.keys())
            all_params = np.empty((len(res.flatchain), len(columns)))
            for i_ind in range(0, len(res.flatchain)):
                for i_c, c in enumerate(columns):
                    all_params[i_ind, i_c] = res.flatchain[c].values[i_ind]
            df_params = pd.DataFrame(data=all_params, columns=columns)
            df_params.to_csv(os.path.join(wd, 'Results', run_ID, 'parameters', 'parameters_emcee.csv'))


            # The values reported in the MinimizerResult are the medians of the probability distributions
            # and a 1 𝜎 quantile, estimated as half the difference between the 15.8 and 84.2 percentiles.
            print('median of posterior probability distribution')
            print('--------------------------------------------')
            lmfit.report_fit(res.params)

            # To obtain the values for the Maximum Likelihood Estimation (MLE) we find the
            # location in the chain with the highest probability
            highest_prob = np.argmax(res.lnprob)
            hp_loc = np.unravel_index(highest_prob, res.lnprob.shape)
            mle_soln = res.chain[hp_loc]
            for i, par in enumerate(pars):
                pars[par].value = mle_soln[i]
            print('\nMaximum Likelihood Estimation from emcee       ')
            print('-------------------------------------------------')
            print('Parameter  MLE Value   Median Value   Uncertainty')
            fmt = '  {:5s}  {:11.5f} {:11.5f}   {:11.5f}'.format
            for name, param in res.params.items():
                print(fmt(name, param.value, res.params[name].value,
                          res.params[name].stderr))

            # Use the samples from emcee to work out the 1- and 2-𝜎 error estimates
            print('\nError estimates from emcee:')
            print('------------------------------------------------------')
            print('Parameter  -2sigma  -1sigma   median  +1sigma  +2sigma 95')
            for name in pars.keys():
                quantiles = np.percentile(res.flatchain[name],[2.275, 15.865, 50, 84.135, 97.275, 95])
                median = quantiles[2]
                err_m2 = quantiles[0] - median
                err_m1 = quantiles[1] - median
                err_p1 = quantiles[3] - median
                err_p2 = quantiles[4] - median
                err_95 = quantiles[5] - median
                fmt = '  {:5s}   {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}'.format
                print(fmt(name, err_m2, err_m1, median, err_p1, err_p2, err_95))


            # Calculate R_eff
            loadFrom = run_ID
            for i_data in i_datalst:
                data  = all_data[i_data]
                keys  = ['R_T1', 'R_T2', 'R_T3']
                names = ['MedInc', 'LivSp', 'Sen']
                n_adm = 3
                R1_abs = []
                R2_abs = []
                R3_abs = []
                R1_eff = []
                R2_eff = []
                R3_eff = []
                for i_ind in range(0, len(res.flatchain)):
                    this_pars = dict()
                    if i_data == 0:
                        this_pars['seed'] = 0
                        this_pars['R_T1'] = res.flatchain['R_T1_MedInc'].values[i_ind]
                        this_pars['R_T2'] = res.flatchain['R_T2_MedInc'].values[i_ind]
                        this_pars['R_T3'] = res.flatchain['R_T3_MedInc'].values[i_ind]
                    elif i_data == 1:
                        this_pars['seed'] = 0
                        this_pars['R_T1'] = res.flatchain['R_T1_Liv'].values[i_ind]
                        this_pars['R_T2'] = res.flatchain['R_T2_Liv'].values[i_ind]
                        this_pars['R_T3'] = res.flatchain['R_T3_Liv'].values[i_ind]
                    elif i_data == 2:
                        this_pars['seed'] = 0
                        this_pars['R_T1'] = res.flatchain['R_T1_Sen'].values[i_ind]
                        this_pars['R_T2'] = res.flatchain['R_T2_Sen'].values[i_ind]
                        this_pars['R_T3'] = res.flatchain['R_T3_Sen'].values[i_ind]

                    for q in range(0, n_adm):
                        norm = all_fixed_pars[i_data][3].values[q, :].sum()
                        if q > 0:
                            R = this_pars[keys[q]] * this_pars[keys[0]]
                        else:
                            R = this_pars[keys[q]]

                        if q == 0:
                            R1_abs.append(R)
                            R1_eff.append(R * norm)
                        elif q == 1:
                            R2_abs.append(R)
                            R2_eff.append(R * norm)
                        else:
                            R3_abs.append(R)
                            R3_eff.append(R * norm)
                df_save = pd.DataFrame(list(zip(R1_abs, R2_abs, R3_abs, R1_eff, R2_eff, R3_eff)),
                                       columns=['R1_abs', 'R2_abs', 'R3_abs', 'R1_eff', 'R2_eff', 'R3_eff'])
                df_save.to_csv(os.path.join(wd, 'Results', run_ID, 'parameters', names[i_data] + '_fitemcee.csv'))

            # Show distributions
            corner.corner(res.flatchain, labels=res.var_names,
                                       truths=list(res.params.valuesdict().values()))
            plt.savefig(os.path.join(wd, 'Results', run_ID, 'figures', 'cornerPlot.png'))
        else:
            loadFrom = run_ID_loadFrom
            df_params = pd.read_csv(
                os.path.join(wd, 'Results', run_ID_loadFrom, 'parameters',
                             'parameters_emcee.csv'))


        # Plot 500 samples from the chains
        n_chains_plot = 500

        # Details for plot
        inds = np.random.randint(df_params.shape[0], size=n_chains_plot)
        colorE = ['black', 'black', 'black']
        colorP = ['darkred', 'darkorange', 'darkblue']
        colorI = ['red', 'gold', 'royalblue']
        colorU = ['lightcoral', 'yellow', 'cornflowerblue']
        titles = ['Median Income', '1P Households', 'Living Space', 'Senior']

        # x ticks:
        dates = np.array(pd.date_range(start, end))
        ts    = pd.to_datetime(dates)
        dates = ts.strftime('%d %b').values
        ticks = np.arange(all_data[0][0][0] + 2, max(all_data[0][0]), 7)

        # Plot cases
        if plotCases:
            for i_data in i_datalst:
                data = all_data[i_data]
                allE = np.empty((len(inds), 61, 3))
                allP = np.empty((len(inds), 61, 3))
                allI = np.empty((len(inds), 61, 3))
                allU = np.empty((len(inds), 61, 3))
                for i_inds, ind in enumerate(inds):

                    # predict for these parameters
                    this_pars = dict()
                    if useTinfUFix:
                        this_pars['TinfUi'] = 2.
                    else:
                        this_pars['TinfUi'] = df_params['TinfUi'].values[ind]
                    this_pars['Tinc']  = df_params['Tinc'].values[ind]
                    this_pars['TinfP'] = df_params['TinfP'].values[ind]
                    this_pars['E0']    = df_params['E0'].values[ind]
                    this_pars['p_unr'] = p_unr_estimate

                    if i_data == 0:
                        the_title = 'Median Income'
                        this_pars['seed'] = 0
                        this_pars['R_T1'] = df_params['R_T1_MedInc'].values[ind]
                        this_pars['R_T2'] = df_params['R_T2_MedInc'].values[ind]
                        this_pars['R_T3'] = df_params['R_T3_MedInc'].values[ind]
                    elif i_data == 1:
                        the_title = 'Living Space'
                        this_pars['seed'] = 1
                        this_pars['R_T1'] = df_params['R_T1_Liv'].values[ind]
                        this_pars['R_T2'] = df_params['R_T2_Liv'].values[ind]
                        this_pars['R_T3'] = df_params['R_T3_Liv'].values[ind]
                    elif i_data == 2:
                        the_title = 'Senior'
                        this_pars['seed'] = 0
                        this_pars['R_T1'] = df_params['R_T1_Sen'].values[ind]
                        this_pars['R_T2'] = df_params['R_T2_Sen'].values[ind]
                        this_pars['R_T3'] = df_params['R_T3_Sen'].values[ind]

                    n_adm = 3
                    # Get solution
                    yfit = solution_SEUI_NEW(data[0], this_pars, all_fixed_pars[i_data]).T

                    thisE = yfit[1 * n_adm:2 * n_adm]
                    allE[i_inds, :, :] = thisE.T

                    thisP = yfit[2 * n_adm:3 * n_adm]
                    allP[i_inds, :, :] = thisP.T

                    thisI = yfit[3 * n_adm:4 * n_adm]
                    allI[i_inds, :, :] = thisI.T

                    thisU = yfit[4 * n_adm:5 * n_adm] + yfit[5 * n_adm:6 * n_adm]
                    allU[i_inds, :, :] = thisU.T

                # Plot results
                for i_nad in range(0, 3):

                    # plot median and 95% confidence bound encompassed by the 2 sigma percentiles
                    fig1 = plt.figure(figsize=(4, 4))
                    ax = fig1.add_subplot(1, 1, 1)
                    plt.plot(data[0], data[i_nad + 1], 'o', c='k', label='Data T' + str(i_nad + 1), markersize=3)


                    # Exposed
                    medianE  = np.median(allE[:, :, i_nad], axis=0)
                    perc97_E = np.percentile(allE[:, :, i_nad], 97.275, axis=0)  # 2.275, 15.865, 50, 84.135, 97.275,
                    perc3_E  = np.percentile(allE[:, :, i_nad], 2.275, axis=0)
                    plt.plot(data[0], medianE, ':', label='E', linewidth=2, zorder=12, color=colorE[i_nad])
                    plt.fill_between(data[0], perc3_E, perc97_E,
                                     alpha=0.25, linewidth=0, zorder=3, color=colorE[i_nad])


                    # Presymptomatic
                    medianP  = np.median(allP[:, :, i_nad], axis=0)
                    perc97_P = np.percentile(allP[:, :, i_nad], 97.275, axis=0)  # 2.275, 15.865, 50, 84.135, 97.275,
                    perc3_P  = np.percentile(allP[:, :, i_nad], 2.275, axis=0)
                    plt.plot(data[0], medianP, '--', label='P', linewidth=2, zorder=12, c=colorP[i_nad])
                    plt.fill_between(data[0], perc3_P, perc97_P,
                                     alpha=0.25, linewidth=0, zorder=3, color=colorP[i_nad])

                    # Infectious, reported but isolated
                    medianI = np.median(allI[:, :, i_nad], axis=0)
                    perc97_I = np.percentile(allI[:, :, i_nad], 97.275, axis=0)  # 2.275, 15.865, 50, 84.135, 97.275,
                    perc3_I = np.percentile(allI[:, :, i_nad], 2.275, axis=0)
                    plt.plot(data[0], medianI, '-', label='I', linewidth=2, zorder=12, c=colorI[i_nad])
                    plt.fill_between(data[0], perc3_I, perc97_I,
                                     alpha=0.25, linewidth=0, zorder=3, color=colorI[i_nad])

                    # Unreported (sum of infectious and recovered)
                    medianU = np.median(allU[:, :, i_nad], axis=0)
                    perc97_U = np.percentile(allU[:, :, i_nad], 97.275, axis=0)  # 2.275, 15.865, 50, 84.135, 97.275,
                    perc3_U = np.percentile(allU[:, :, i_nad], 2.275, axis=0)
                    plt.plot(data[0], medianU, '-.', label='U_i+U_r', linewidth=2, zorder=12, c=colorU[i_nad])
                    plt.fill_between(data[0], perc3_U, perc97_U,
                                     alpha=0.25, linewidth=0, zorder=3, color=colorU[i_nad])

                    # Make plot pretty
                    plt.legend(prop={'size': 10}, frameon=False, loc="upper left")
                    plt.yscale('log')
                    plt.yticks(fontsize=12)
                    plt.xticks(fontsize=12)
                    plt.ylim([1, 10000])  # 1.2*maxN])
                    plt.xlim([0, len(data[0]) - 1])
                    plt.ylabel('Number of Cases', fontsize=14)
                    ax.set_xticks(ticks)
                    int_ticks = [int(i) for i in ticks]
                    ax.set_xticklabels(dates[int_ticks], rotation=45, ha='right')
                    ax.patch.set_facecolor('lightgrey')
                    plt.grid(color='white')
                    plt.tight_layout()

                    # Save
                    os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
                    fig1.savefig(titles[i_data] + '_cases_T' + str(i_nad + 1) + '.png', dpi=250)
                    fig1.savefig(titles[i_data] + '_cases_T' + str(i_nad + 1) + '.pdf', format='pdf')
                    os.chdir(wd)

        # Plot effective reproductive number over time
        if plotR:
            names  = ['MedInc', 'LivSp', 'Sen']
            keys   = ['R1_eff', 'R2_eff', 'R3_eff']
            labels = ['T1', 'T2', 'T3']
            colorI = ['red', 'black', 'blue']

            # Plot Reff over Time
            for i_data in i_datalst:
                time_dep  = all_fixed_pars[i_data][1]
                t         = np.arange(61)
                alpha_soc = time_dep_soc(t)
                alpha_mob = time_dep(t)

                df_save   = pd.read_csv(os.path.join(wd, 'Results', loadFrom, 'parameters',
                                 names[i_data] + '_fitemcee.csv'))

                for i_nad in range(0, 3):
                    med_R = np.percentile(df_save[keys[i_nad]], 50) * alpha_soc * alpha_mob
                    min_R = np.percentile(df_save[keys[i_nad]], 2.275) * alpha_soc * alpha_mob
                    max_R = np.percentile(df_save[keys[i_nad]], 97.275) * alpha_soc * alpha_mob

                    # plot time series
                    fig2 = plt.figure(figsize=(4, 4))
                    ax = fig2.add_subplot(1, 1, 1)

                    plt.plot(t, med_R, '-', label=labels[i_nad], linewidth=2, zorder=12,
                             color=colorI[i_nad])
                    plt.fill_between(data[0], min_R, max_R,
                                     alpha=0.25, linewidth=0, zorder=3, color=colorI[i_nad])

                    plt.legend(prop={'size': 10}, frameon=False, loc="upper right")

                    plt.yticks(fontsize=12)
                    plt.xticks(fontsize=12)
                    # plt.ylim([1, 150])  # 1.2*maxN])
                    plt.xlim([0, len(data[0]) - 1])
                    plt.ylabel('$R_{eff}$', fontsize=14)
                    ax.set_xticks(ticks)
                    int_ticks = [int(i) for i in ticks]
                    ax.set_xticklabels(dates[int_ticks], rotation=45, ha='right')
                    ax.patch.set_facecolor('lightgrey')
                    plt.grid(color='white')
                    plt.tight_layout()

                    os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
                    fig2.savefig(titles[i_data] + '_ReffT' + str(i_nad + 1) + '.png', dpi=250)
                    fig2.savefig(titles[i_data] + '_ReffT' + str(i_nad + 1) + '.pdf', format='pdf')
                    os.chdir(wd)

            # Plot Histograms
            for i_data in i_datalst:
                df_save = pd.read_csv(
                    os.path.join(wd, 'Results', loadFrom, 'parameters',
                                 names[i_data] + '_fitemcee.csv'))

                # plot time series
                fig2 = plt.figure(figsize=(4, 4))
                ax = fig2.add_subplot(1, 1, 1)
                for i_nad in range(0, 3):
                    plt.hist(df_save[keys[i_nad]], color=colorI[i_nad], bins=20, alpha=0.5, label=labels[i_nad])
                    plt.vlines(np.percentile(df_save[keys[i_nad]], 50), 0, 7000, linestyles="solid",
                               colors=colorI[i_nad], linewidth=2)
                    plt.vlines(np.percentile(df_save[keys[i_nad]], 2.275), 0, 7000, linestyles="dashed",
                               colors=colorI[i_nad], linewidth=2)
                    plt.vlines(np.percentile(df_save[keys[i_nad]], 97.275), 0, 7000, linestyles="dashed",
                               colors=colorI[i_nad], linewidth=2)

                plt.legend(prop={'size': 10}, frameon=False, loc="upper right")

                plt.yticks([])
                plt.xticks(fontsize=12)
                # plt.ylim([1, 150])  # 1.2*maxN])
                plt.xlim([1, 3.5])
                plt.ylim([0, 7000])
                plt.ylabel('Number of realizations', fontsize=14)
                plt.xlabel('$R_{eff}(0)$', fontsize=14)
                ax.patch.set_facecolor('lightgrey')
                plt.grid(color='white')
                plt.tight_layout()

                os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
                fig2.savefig(titles[i_data] + '_Reff_hist.png', dpi=250)
                fig2.savefig(titles[i_data] + '_Reff_hist.pdf', format='pdf')
                os.chdir(wd)

        # Predict mobility
        if plot_mobility:
            global zeroMobility
            global constantMobility
            n_adm = 3
            for i_data in i_datalst:
                data = all_data[i_data]
                all_mob_actual = np.empty((len(inds), 61))
                all_constmob = np.empty((len(inds), 61))
                for i_inds, ind in enumerate(inds):

                    # predict for these parameters
                    this_pars = dict()
                    if useTinfUFix:
                        this_pars['TinfUi'] = 2.
                    else:
                        this_pars['TinfUi'] = df_params['TinfUi'].values[ind]
                    this_pars['Tinc']  = df_params['Tinc'].values[ind]
                    this_pars['TinfP'] = df_params['TinfP'].values[ind]
                    this_pars['E0']    = df_params['E0'].values[ind]
                    this_pars['p_unr'] = p_unr_estimate

                    if i_data == 0:
                        the_title = 'Median Income'
                        this_pars['seed'] = 0
                        this_pars['R_T1'] = df_params['R_T1_MedInc'].values[ind]
                        this_pars['R_T2'] = df_params['R_T2_MedInc'].values[ind]
                        this_pars['R_T3'] = df_params['R_T3_MedInc'].values[ind]
                    elif i_data == 1:
                        the_title = '1P-Household'
                        this_pars['seed'] = 1
                        this_pars['R_T1'] = df_params['R_T1_Liv'].values[ind]
                        this_pars['R_T2'] = df_params['R_T2_Liv'].values[ind]
                        this_pars['R_T3'] = df_params['R_T3_Liv'].values[ind]
                    elif i_data == 2:
                        the_title = 'Senior'
                        this_pars['seed'] = 0
                        this_pars['R_T1'] = df_params['R_T1_Sen'].values[ind]
                        this_pars['R_T2'] = df_params['R_T2_Sen'].values[ind]
                        this_pars['R_T3'] = df_params['R_T3_Sen'].values[ind]

                    zeroMobility     = False
                    constantMobility = False
                    fit = solution_SEUI_NEW(data[0], this_pars, all_fixed_pars[i_data]).T
                    constantMobility = True
                    fit_const = solution_SEUI_NEW(data[0], this_pars, all_fixed_pars[i_data]).T
                    zeroMobility = False
                    constantMobility = False

                    thisU = fit[4 * n_adm:5 * n_adm] + fit[5 * n_adm:6 * n_adm] + fit[3 * n_adm:4 * n_adm]
                    all_mob_actual[i_inds, :] = np.sum(thisU, axis=0)

                    thisU = fit_const[4 * n_adm:5 * n_adm] + fit_const[5 * n_adm:6 * n_adm] + fit_const[
                                                                                              3 * n_adm:4 * n_adm]
                    all_constmob[i_inds, :] = np.sum(thisU, axis=0)


                fig2 = plt.figure(figsize=(4, 4))
                ax = fig2.add_subplot(1, 1, 1)

                std_mob_actual = np.std(all_mob_actual, axis=0)
                median_mob_actual = np.median(all_mob_actual, axis=0)
                perc97_mob_actual = np.percentile(all_mob_actual, 97.275, axis=0)  # 2.275, 15.865, 50, 84.135, 97.275,
                perc3_mob_actual = np.percentile(all_mob_actual, 2.275, axis=0)
                plt.plot(data[0], median_mob_actual, '-', label='M0-observed', linewidth=2, zorder=12, c='k')
                plt.fill_between(data[0], perc3_mob_actual, perc97_mob_actual,
                                 alpha=0.25, linewidth=0, zorder=3, color='k')

                std_constmob = np.std(all_constmob, axis=0)
                median_constmob = np.median(all_constmob, axis=0)
                perc97_constmob = np.percentile(all_constmob, 97.275, axis=0)  # 2.275, 15.865, 50, 84.135, 97.275,
                perc3_constmob = np.percentile(all_constmob, 2.275, axis=0)
                plt.plot(data[0], median_constmob, ':', label='M1-const. mobility', linewidth=2, zorder=12,
                         color='blue')
                plt.fill_between(data[0], perc3_constmob, perc97_constmob,
                                 alpha=0.25, linewidth=0, zorder=3, color='blue')


                plt.legend(prop={'size': 10}, frameon=False, loc="upper left")
                plt.yscale('log')
                plt.yticks(fontsize=12)
                plt.xticks(fontsize=12)
                plt.xlim([0, len(data[0]) - 1])
                plt.yscale('linear')
                plt.ylabel('Number of Cases', fontsize=14)
                ax.set_xticks(ticks)
                int_ticks = [int(i) for i in ticks]
                ax.set_xticklabels(dates[int_ticks], rotation=45, ha='right')
                ax.patch.set_facecolor('lightgrey')
                plt.grid(color='white')
                plt.tight_layout()

                os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
                fig2.savefig(titles[i_data] + '_mobility.png', dpi=250)
                fig2.savefig(titles[i_data] + '_mobility.pdf', format='pdf')
                os.chdir(wd)

        # Predict vaccination scenarios
        if plot_vaccine_all:

            # Plot based on median incom partition (i_data = 0), for living space: i_data = 1, seniority: i_data = 2
            eval_vaccine_all_EMCEE(df_params, all_fixed_pars, n_chains_plot, titles, i_data =0)

        if plot_vaccine_individ:
            eval_VACCINE_EMCEE(df_params, all_fixed_pars, n_chains_plot, i_datalst)



    return 0

### FUNCTIONS #################################################################

# Data preparation
def get_mobilityTimeSeries():

    # Load time series
    delta_t_mobility      = (start_mobility - start).days
    df_timedep            = pd.read_csv(filename_mobility)
    n_travelling          = df_timedep['total'].values[:-2]

    # Get the correct time frame
    time     = 7 * df_timedep.index.values[:-2] + delta_t_mobility
    time_dep = UnivariateSpline(time, n_travelling)
    time_dep.set_smoothing_factor(0.0001)

    # x-Axis labels
    dates     = np.array(pd.date_range(start, end))
    t         = np.arange(len(dates))
    ts        = pd.to_datetime(dates)
    dates     = ts.strftime('%m.%d').values
    ticks     = np.arange(min(t), max(t), 7)
    int_ticks = [int(i) for i in ticks]

    # actual plot
    fig1 = plt.figure(figsize=(3, 3))
    ax   = fig1.add_subplot()
    plt.plot(t, time_dep(t), label = 'Mobility time series',color = 'teal')
    plt.ylabel('Mobility Interaction Score')
    plt.yticks(fontsize=10)
    ax.set_xticks(ticks)
    ax.set_xticklabels(dates[int_ticks], rotation=45,fontsize=10)
    plt.tight_layout()

    df_mobility = pd.DataFrame(data = np.array([dates,t,time_dep(t)]).T, index = dates,columns =['date','time', 'alpha_mob'])
    df_mobility.to_csv(os.path.join(wd,'Results',run_ID,'parameters','alpha_mob.csv'))
    return time_dep
def get_SocialInteractionTimeSeries(time_dep):

    # Get social time series
    df_Kalman   = pd.read_csv(kalmanFilterResult)
    dates       = np.array(pd.date_range(start, end))
    t           = np.arange(0,len(dates))
    R_estimate  = df_Kalman['R_estimate'].values

    # Account for set-off between dates to consider and time series for kalman filter
    delta_t_kalman = (start_kalman-start).days
    if delta_t_kalman >0:
        R_estimate  = np.array(list(R_estimate[0]*np.ones(delta_t_kalman,))+list(R_estimate))
    elif delta_t_kalman <0:
        raise('Social interaction time series not covered - ensure no offset to start date')

    # Divide by mobility time series to end up with social interaction changes
    alpha_mob     = time_dep(t)
    y_Kalman      = R_estimate /alpha_mob
    y_soc         = y_Kalman / np.max(y_Kalman)
    time_dep_soc  = UnivariateSpline(t, y_soc, s = 0.03)


    # X-Axis labels
    dates     = np.array(pd.date_range(start, end))
    ts        = pd.to_datetime(dates)
    dates     = ts.strftime('%m.%d').values
    ticks     = np.arange(min(np.arange(0, 61)), max(np.arange(0, 61)), 7)
    int_ticks = [int(i) for i in ticks]

    # Actual plot
    fig1 = plt.figure(figsize=(3, 3))
    ax = fig1.add_subplot()
    plt.plot(t, R_estimate, label='estimated R',color = 'teal')
    plt.plot(t, y_Kalman, label = 'R/' + r'$\alpha_{mob}$',color = 'darkblue')
    plt.plot(t, y_soc, '--', label = 'Normalized',color = 'black',zorder = 3)
    plt.plot(t, time_dep_soc(t), label = 'Smoothend',color = 'red')
    plt.ylabel('Social Interaction Score',fontsize=10)
    ax.set_xticks(ticks)
    plt.yticks(fontsize=8)
    ax.set_xticklabels(dates[int_ticks], rotation=45,fontsize=8)
    plt.legend(frameon=False,loc="upper right",fontsize=8)
    plt.tight_layout()


    # Save
    df_soc = pd.DataFrame(data=np.array([t, time_dep_soc(t),y_soc,y_Kalman,R_estimate]).T, index=dates,
                          columns=['time', 'alpha_soc','R/alpha_mob norm','R/alpha_mob','R_kalman'])
    df_soc.to_csv(os.path.join(wd, 'Results',run_ID,'parameters', 'alpha_soc.csv'))

    return time_dep_soc
def get_MobilityMatrix():


    transport_means = ['publ', 'bike', 'moto', 'foot']
    n_splitsSoc     = 3
    quarter         = [i+1 for i in range(0,n_splitsSoc)]


    all_A = []
    if useRandomTiles:
        partitions_MobilityMatrix = [mobilityMatrix_random_prefix]
    else:
        partitions_MobilityMatrix= [mobilityMatrix_MedInc_prefix,
                                    mobilityMatrix_seniority_prefix,
                                    mobilityMatrix_LivSpace_prefix]

    for partition in partitions_MobilityMatrix:
        for i, tr in enumerate(transport_means):

            file = os.path.join(wd, 'graphs', partition + tr + '_mobility.csv')
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
        all_A.append(A)

    return all_A

# Model, solution, and residuals
def ode_model(Y, t, pars, fixed_pars):
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

    n_adm        = fixed_pars[0]  # number of tiles
    time_dep     = fixed_pars[1]  # mobility time series
    time_dep_soc = fixed_pars[2]  # mobility time series
    Adj          = fixed_pars[3]  # Adjacency matrix - effective connectivity

    # Model parameters
    R_infU_in = [pars['R_T1'], pars['R_T2'], pars['R_T3']]
    T_inc     = pars['Tinc']
    T_infU    = pars['TinfP']
    p_unr     = pars['p_unr']
    if useTinfUFix:
        T_inf_Ui = 2
    else:
        T_inf_Ui = pars['TinfUi']


    # Mobility and social interaction time series
    alpha_mob = time_dep(t)
    alpha_soc = time_dep_soc(t)

    # Time dependence for measures taken on R - social interaction time series
    R_infU = np.array(R_infU_in)
    for i_tile in range(1, n_adm):
        R_infU[i_tile] = R_infU[0] * R_infU_in[i_tile]

    # Options for scenario simulations
    if constantMobility:
        alpha_mob = 1

    if constantR:
        alpha_soc = fixedSocial

    # time variation
    alpha_use = alpha_mob*alpha_soc

    # Same reproductive number for U and U_i
    R_inf_Ui = R_infU

    # compartments
    s = Y[0 * n_adm:1 * n_adm]
    e = Y[1 * n_adm:2 * n_adm]
    u = Y[2 * n_adm:3 * n_adm]
    i = Y[3 * n_adm:4 * n_adm]
    u_i = Y[4 * n_adm:5 * n_adm]
    u_r = Y[5 * n_adm:6 * n_adm]
    n = s + e + i + u +u_i + u_r


    # Susceptibles: Add - diffusion term to E - diffusion term to A
    dsdt = - np.multiply(alpha_use * s * R_infU / (n), np.dot(Adj, np.multiply(1. / T_infU, u))) \
           - np.multiply(alpha_use * s * R_inf_Ui / (n), np.dot(Adj, np.multiply(1. / T_inf_Ui, u_i)))

    # Exposed - not infectious
    dedt = - np.multiply(1 / T_inc, e) \
           + np.multiply(alpha_use * s * R_infU / (n), np.dot(Adj, np.multiply(1. / T_infU, u))) \
           + np.multiply(alpha_use * s * R_inf_Ui / (n), np.dot(Adj, np.multiply(1. / T_inf_Ui, u_i)))

    dudt = np.multiply(1 / T_inc, e) - np.multiply(1 / T_infU, u)

    # Reported infected - assumed to be isolated
    didt = (1 - p_unr) * np.multiply(1 / T_infU, u)


    # Unreported infected - infectious and not isolated (might know about symptoms, hence different R from U compartment)
    du_idt = p_unr * np.multiply(1 / T_infU, u) - np.multiply(1 / T_inf_Ui, u_i)

    # Unreported recovered - account for duration of infectious periode
    du_rdt = np.multiply(1 / T_inf_Ui, u_i)



    # output has to have shape
    output = np.concatenate((dsdt, dedt, dudt, didt, du_idt, du_rdt))

    return output
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

    n_adm        = fixed_pars[0] # number of tiles
    time_dep     = fixed_pars[1] # mobility time series
    time_dep_soc = fixed_pars[2] # mobility time series
    Adj          = fixed_pars[3] # Adjacency matrix - effective connectivity

    # Model parameters
    R_infU_in = pars[:n_adm]
    T_inc     = pars[n_adm]
    T_infU    = pars[4 * n_adm + 1]
    T_inf_Ui  = pars[5 * n_adm + 4]
    p_unr     = pars[5 * n_adm + 5]

    # Mobility and social interaction time series
    alpha_mob = time_dep(t)
    alpha_soc = time_dep_soc(t)

    # R is a relative parameter here
    R_infU = np.array(R_infU_in)
    for i_tile in range(1, n_adm):
        R_infU[i_tile] = R_infU[0] * R_infU_in[i_tile]

    # Options for scenario simulations
    if constantMobility:
        alpha_mob = 1

    if constantR:
        alpha_soc = fixedSocial

    # Mobility and social interaction time series
    alpha_use = alpha_mob* alpha_soc

    # Same reproductive number for U and U_i
    R_inf_Ui = R_infU

    # compartments
    s = Y[0 * n_adm:1 * n_adm]
    e = Y[1 * n_adm:2 * n_adm]
    u = Y[2 * n_adm:3 * n_adm]
    i = Y[3 * n_adm:4 * n_adm]
    u_i = Y[4 * n_adm:5 * n_adm]
    u_r = Y[5 * n_adm:6 * n_adm]
    n = s + e + i + u +u_i + u_r


    # Susceptibles: Add - diffusion term to E - diffusion term to A
    dsdt = - np.multiply(alpha_use * s * R_infU / (n), np.dot(Adj, np.multiply(1. / T_infU, u))) \
           - np.multiply(alpha_use * s * R_inf_Ui / (n), np.dot(Adj, np.multiply(1. / T_inf_Ui, u_i)))

    # Exposed - not infectious
    dedt = - np.multiply(1 / T_inc, e) \
           + np.multiply(alpha_use * s * R_infU / (n), np.dot(Adj, np.multiply(1. / T_infU, u))) \
           + np.multiply(alpha_use * s * R_inf_Ui / (n), np.dot(Adj, np.multiply(1. / T_inf_Ui, u_i)))

    dudt = np.multiply(1 / T_inc, e) - np.multiply(1 / T_infU, u)

    # Reported infected - assumed to be isolated
    didt = (1 - p_unr) * np.multiply(1 / T_infU, u)


    # Unreported infected - infectious and not isolated (might know about symptoms, hence different R from U compartment)
    du_idt = p_unr * np.multiply(1 / T_infU, u) - np.multiply(1 / T_inf_Ui, u_i)

    # Unreported recovered - account for duration of infectious periode
    du_rdt = np.multiply(1 / T_inf_Ui, u_i)



    # output has to have shape
    output = np.concatenate((dsdt, dedt, dudt, didt, du_idt, du_rdt))

    return output

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
    n_cmp = 6
    s_0   = fixed_pars[4]
    y0    = np.concatenate((s_0, par_list[:((n_cmp - 1) * n_adm)]))
    pars  = par_list[(n_cmp - 1) * n_adm:]
    sol   = odeint(lambda a, b, c: ode_model_SEUI_SymMat(a, b, c, fixed_pars), y0, t, args=(pars,))

    return sol
def solution_SEUI_NEW(t, pars, fixed_pars):
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

    n_adm = fixed_pars[0] # number of subsets in the partition
    n_cmp = 6             # number of compartments - here always 6

    # Initial conditions - all zero apart from susceptibles
    s_0 = fixed_pars[4]
    y0  = np.concatenate((s_0, np.repeat(0, n_adm * (n_cmp - 1))))

    # Use a single seed in compartment E
    y0[n_adm + pars['seed']] = pars['E0']

    # Get ODE solution for this set of parameters
    sol = odeint(lambda a, b, c: ode_model(a, b, c, fixed_pars), y0, t, args=(pars,))

    return sol

def residual_SEUI_NEW(all_pars, data, fixed_pars):
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

    n_adm    = fixed_pars[0]
    t        = data[0]
    n_cuminf = data[1:n_adm + 1, :]

    # specific parameters
    pars = dict()
    pars['TinfUi']= all_pars['TinfUi']
    pars['Tinc']  = all_pars['Tinc']
    pars['TinfP'] = all_pars['TinfP']
    pars['E0']    = all_pars['E0']
    pars['R_T1']  = all_pars['R_T1']
    pars['R_T2']  = all_pars['R_T2']
    pars['R_T3']  = all_pars['R_T3']
    pars['seed']  = 0
    pars['p_unr'] = p_unr_estimate

    # Solution with given parameter set
    yfit = solution_SEUI_NEW(t, pars, fixed_pars).T
    yfit_cuminf = yfit[3 * n_adm:4 * n_adm]

    # Calculate residuals
    resid = 0.0 * n_cuminf
    for i in range(0, n_adm):
        inds = np.logical_and(~np.isinf(np.gradient(n_cuminf[i, :])),
                              ~np.isinf(np.gradient(yfit_cuminf[i, :])))
        yfit_cuminf_use = yfit_cuminf[i, inds]
        n_cuminf_use = n_cuminf[i, inds]
        resid[i, :] = np.gradient(yfit_cuminf_use) - np.gradient(n_cuminf_use)

    return resid.flatten()
def residual_SEUI_NEW_parallel(all_pars, all_data, all_fixed_pars):
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

    n_adm = all_fixed_pars[0][0]
    resid = np.zeros((3 * n_adm, len(all_data[0][0])))
    counter = 0
    for i_data in [0, 1, 2]:
        data     = all_data[i_data]
        t        = data[0]
        n_cuminf = data[1:n_adm + 1, :]
        pars     = dict()

        if useTinfUFix:
            pars['TinfUi'] = 2.
        else:
            pars['TinfUi'] = all_pars['TinfUi']
        pars['Tinc']       = all_pars['Tinc']
        pars['TinfP']      = all_pars['TinfP']
        pars['E0']         = all_pars['E0']
        pars['fixed_pars'] = all_fixed_pars[i_data]
        pars['p_unr']      = p_unr_estimate

        if i_data == 0:
            pars['R_T1'] = all_pars['R_T1_MedInc']
            pars['R_T2'] = all_pars['R_T2_MedInc']
            pars['R_T3'] = all_pars['R_T3_MedInc']
            pars['seed'] = 0
        elif i_data == 1:
            pars['R_T1'] = all_pars['R_T1_Liv']
            pars['R_T2'] = all_pars['R_T2_Liv']
            pars['R_T3'] = all_pars['R_T3_Liv']
            pars['seed'] = 0
        elif i_data == 2:
            pars['R_T1'] = all_pars['R_T1_Sen']
            pars['R_T2'] = all_pars['R_T2_Sen']
            pars['R_T3'] = all_pars['R_T3_Sen']
            pars['seed'] = 0

        # Solution with given parameter set
        yfit = solution_SEUI_NEW(t, pars, all_fixed_pars[i_data]).T

        # Cummulative case reported case numbers - compartment I
        yfit_cuminf = yfit[3 * n_adm:4 * n_adm]

        # Get absolute numbers and calcualte residuals
        for i in range(0, n_adm):
            inds = np.logical_and(~np.isinf(np.gradient(n_cuminf[i, :])),
                                  ~np.isinf(np.gradient(yfit_cuminf[i, :])))
            yfit_cuminf_use = yfit_cuminf[i, inds]
            n_cuminf_use = n_cuminf[i, inds]
            resid[counter, :] = np.gradient(yfit_cuminf_use) - np.gradient(n_cuminf_use)
            counter = counter + 1

    return resid.flatten()


# Predict vaccine scenarios
def eval_vaccine_all_EMCEE(df_params, all_fixed_pars, n_chains, names_data, i_data =0):
    print('Plotting vaccine scenario - vaccination at random using different vaccine properties and fraction vaccinated!')

    # Scenario details
    icu_stay       = 5.9          # Median duration of ICU stay
    icu_perc       = 1 * 0.01     # Percentage of all cases needing icu treatment
    icu_capa       = 44           # ICU bed capacity
    fractions_vacc = [0, 0.3333333, 0.6666666]
    effVacs        = [0.9, 0.6]

    # plot details
    colorsMean = ['black', 'darkblue', 'royalblue', 'darkred', 'crimson']
    tvac       = np.arange(0, 150) # time range simulated
    randInds   = np.random.randint(df_params.shape[0], size=n_chains)
    fig1 = plt.figure(figsize=(4, 4))
    ax = fig1.add_subplot(1, 1, 1)

    # general settings
    all_quat = [1, 2, 3]
    counter  = 0

    # Loop over all options
    for ev in range(0, len(effVacs)):
        effVac = effVacs[ev]

        for i_fv, frac_vac in enumerate(fractions_vacc):

            if frac_vac == 0 and ev > 0:
                continue

            all_ninf = np.zeros((n_chains, len(tvac)))

            for j in range(0, n_chains):
                ind = randInds[j]
                this_params = getPar_list(df_params, ind, i_data)
                for q in all_quat:
                    if q == all_quat[0]:
                        mdl_cuminf_uncer = predict_Vaccine(this_params, all_fixed_pars[i_data], frac_vac, tvac, q,
                                                           effVac)
                    else:
                        mdl_cuminf_uncer = mdl_cuminf_uncer + predict_Vaccine(this_params, all_fixed_pars[i_data],
                                                                              frac_vac,
                                                                              tvac, q, effVac)
                all_ninf[j, :] = mdl_cuminf_uncer

            # Get median and 95% confidence bounds
            mean_ninf = np.median(all_ninf, axis=0)
            min_ninf  = np.percentile(all_ninf, 2.275, axis=0)
            max_ninf  = np.percentile(all_ninf, 97.275, axis=0)
            if frac_vac == 0:
                ax.plot(tvac, mean_ninf, '-', linewidth=2, color=colorsMean[counter], zorder=12, label='V0')
            else:
                if i_fv == 1:
                    ax.plot(tvac, mean_ninf, '-', linewidth=2, color=colorsMean[counter], zorder=12,
                            label='V1, ' + f"{frac_vac * 100:.0f}" + '% ' + ' ' + f"{effVac * 100:.0f}" + '% eff.')
                else:
                    ax.plot(tvac, mean_ninf, '--', linewidth=2, color=colorsMean[counter], zorder=12,
                            label='V1, ' + f"{frac_vac * 100:.0f}" + '% ' + ' ' + f"{effVac * 100:.0f}" + '% eff.')

            try:

                yV0_int = interp1d(tvac, mean_ninf)
                cases = []
                for it in range(0, len(tvac) - int(round(icu_stay))):
                    cases.append((yV0_int(it + icu_stay) - yV0_int(it)) * icu_perc)

                yV0_int_cases = interp1d(100 * np.array(cases)[:cases.index(max(cases))] / icu_capa,
                                         np.arange(cases.index(max(cases))))

                xV0 = yV0_int_cases(50)
                yV0_int = interp1d(tvac, mean_ninf)
                yV0 = yV0_int(xV0)

                plt.scatter(xV0, yV0, color=colorsMean[counter], marker='o', s=50, zorder=20)
            except:
                print('50percent ICU capcity never reached!')
            ax.fill_between(tvac, min_ninf, max_ninf, color=colorsMean[counter], zorder=1, alpha=0.25,
                            linewidth=0)
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

    os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
    fig1.savefig(names_data[i_data] + 'vaccineAll.png', dpi=250)
    fig1.savefig(names_data[i_data] + 'vaccineAll.pdf', format='pdf')
    os.chdir(wd)
    return 0
def eval_VACCINE_EMCEE(df_params, all_fixed_pars, n_chains, i_datalst):
    names = ['Median income', 'Living space\nper person', 'Seniority']

    for i_data in [0]:
        output = evalVaccination_new_ICUprotection_EMCEE([i_data], [names[i_data]],
                                                         df_params, all_fixed_pars[i_data], n_chains=n_chains)
        output_icu = evalVaccination_new_ICUprotection_ICU_EMCEE([i_data], [names[i_data]],
                                                                 df_params, all_fixed_pars[i_data], n_chains=n_chains)

    # V0,V1,V2,V3
    combinerun_IDs = [2, 0]
    names_combi    = ['Seniority', 'Median income']
    names_legend   = ['Seniority', 'Median\nincome']
    fractions_vaccGroup = [0.0001]  # [0.0001]#
    fractions_vaccGroup0 = [1.]
    evalVaccination_new_ICUprotection_CombiSenMedInc_EMCEE([2, 0], names_combi, names_legend, output, df_params,
                                                           all_fixed_pars[combinerun_IDs[1]], fractions_vaccGroup,
                                                           fractions_vaccGroup0, n_chains=n_chains)
    evalVaccination_new_ICUprotection_CombiSenMedInc_ICU_EMCEE([2, 0], names_combi, names_legend, output_icu, df_params,
                                                               all_fixed_pars[combinerun_IDs[1]], fractions_vaccGroup,
                                                               fractions_vaccGroup0, n_chains=n_chains)

    # V0,V1,V4
    combinerun_IDs = [2, 0]
    names_combi    = ['Seniority', 'Median income']
    names_legend   = ['Seniority', 'Median\nincome']
    fractions_vaccGroup  = [0.115]
    fractions_vaccGroup0 = [0.5]
    evalVaccination_new_ICUprotection_CombiSenMedInc_EMCEE([2, 0], names_combi, names_legend, None, df_params,
                                                           all_fixed_pars[combinerun_IDs[1]], fractions_vaccGroup,
                                                           fractions_vaccGroup0, n_chains=n_chains)
    evalVaccination_new_ICUprotection_CombiSenMedInc_ICU_EMCEE([2, 0], names_combi, names_legend, None, df_params,
                                                               all_fixed_pars[combinerun_IDs[1]], fractions_vaccGroup,
                                                               fractions_vaccGroup0, n_chains=n_chains)

    return 0
def evalVaccination_new_ICUprotection_EMCEE(i_data_lst, names, df_params, fixed_pars, n_chains=100):

    # general setup
    t                   = list(np.arange(0, 150))
    fractions_vaccGroup = [0.23]
    effVacsGroup        = [0.9]
    effVacsGroup_Sick   = 0.9
    icu_capa            = 44
    icu_stay            = 5.9
    vaccineGroup        = [1, 1, 3] # Vaccinate med inc T1, living space T1, seniority T3
    icu_perc = 1 * 0.01
    if names[0] == 'Seniority':
        vaccinateALLsenior = True
    else:
        vaccinateALLsenior = False
    if vaccinateALLsenior:
        icu_perc2 = 0.005
    else:
        icu_perc2 = 1 * 0.01


    output   = {}
    randInds = np.random.randint(df_params.shape[0], size=n_chains)
    n_adm = 3
    all_quat = [1, 2, 3]  # the summarized quaters of Basel to be analysed: choose from 1-9,'all'
    colors = ['black', 'darkblue', 'gold', 'royalblue', 'orange', 'yellow']


    global constantR
    global constantMobility
    global zeroMobility


    # Get populations shared between the two groups

    # Get all population data
    pop_df = pd.read_csv(filenameSocioeconomicData)


    # Get socioeconomic data
    soc_df0 = get_socDF('Seniority')
    soc_df1 = get_socDF(names[0])

    # For each tile get the blocks and population
    blocks0 = soc_df0['BLO_ID'].loc[soc_df0['percentile'] == 3].values

    # Get population of group 0 in the tiles of group 1
    pop = []
    for the_quarter in [1, 2, 3]:
        blocks1 = soc_df1['BLO_ID'].loc[soc_df1['percentile'] == the_quarter].values

        # Overlapping plocks of group 0 and these blocks
        blocks = list(set(blocks0).intersection(blocks1))

        pop.append(pop_df['Population 2017'].loc[pop_df['BlockID'].isin(blocks)].sum())

    fig1 = plt.figure(figsize=(4, 4))
    ax = fig1.add_subplot(1, 1, 1)
    counter = 0
    for i_q in i_data_lst:

        # set up
        n_tot = np.sum(fixed_pars[4])
        n_vaccineGroup = fixed_pars[4][vaccineGroup[i_q] - 1]
        frac_notSick = effVacsGroup_Sick * fractions_vaccGroup[0]
        n_notsick = frac_notSick * n_tot
        f25 = fractions_vaccGroup[0] * n_tot / n_vaccineGroup
        fractions_vacc_group = [f25]
        fractions_vacc_all = [0, fractions_vaccGroup[0]]

        for ev in range(0, len(effVacsGroup)):
            effVac = effVacsGroup[ev]
            n_toremoved = fractions_vaccGroup[0] * n_tot * effVac
            n_notSickremain = n_notsick - n_toremoved
            n_s0 = n_tot - n_toremoved
            perc_notsick = n_notSickremain / n_s0

            # Redefine ICU percentage
            icu_perc_use = icu_perc2 * (1 - perc_notsick)
            icu_perc_rand = icu_perc * (1 - perc_notsick)

            if counter == 0:

                # random distribution of vaccinated subjects
                for i_fv, frac_vac in enumerate(fractions_vacc_all):

                    if frac_vac == 0 and ev > 0:
                        continue

                    all_ninf = np.zeros((n_chains, len(t)))
                    for j in range(0, n_chains):
                        ind = randInds[j]
                        this_params = getPar_list(df_params, ind, i_q)
                        for q in all_quat:
                            if q == all_quat[0]:
                                mdl_cuminf_uncer = predict_Vaccine(this_params, fixed_pars, frac_vac, t, q, effVac)
                            else:
                                mdl_cuminf_uncer = mdl_cuminf_uncer + predict_Vaccine(this_params, fixed_pars, frac_vac,
                                                                                      t, q, effVac)

                        all_ninf[j, :] = mdl_cuminf_uncer

                    # Average over boot straps
                    mean_ninf = np.median(all_ninf, axis=0)
                    min_ninf = np.percentile(all_ninf, 2.275, axis=0)
                    max_ninf = np.percentile(all_ninf, 97.275, axis=0)
                    # min_ninf = mean_ninf - 2*np.std(all_ninf, axis=0)
                    # max_ninf = mean_ninf + 2*np.std(all_ninf, axis=0)
                    if frac_vac == 0:

                        yV0_int = interp1d(t, mean_ninf)
                        cases = []
                        for it in range(0, len(t) - int(round(icu_stay))):
                            cases.append((yV0_int(it + icu_stay) - yV0_int(it)) * icu_perc_rand)

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
                            cases.append((yV1_int(it + icu_stay) - yV1_int(it)) * icu_perc_rand)

                        yV1_int_cases = interp1d(100 * np.array(cases)[:cases.index(max(cases))] / icu_capa,
                                                 np.arange(cases.index(max(cases))))

                        xV1 = yV1_int_cases(50)
                        yV1_int = interp1d(t, mean_ninf)
                        yV1 = yV1_int(xV1)

                        # ax.plot(np.arange(len(cases)), 100*np.array(cases)/icu_capa, '-', linewidth=2, color=colors[counter], zorder=12,
                        #         label='V1 - ' + f"{frac_vac * 100:.0f}" + '% randomly')  # with '+ f"{effVac*100:.0f}"+'% efficacy')
                        ax.plot(t, mean_ninf, '-', linewidth=2, color=colors[counter], zorder=12,
                                label='V1 - ' + f'{frac_vac * 100:.0f}' + '% randomly')

                    cases_min = []
                    cases_max = []
                    yV0_int_min = interp1d(t, min_ninf)
                    yV0_int_max = interp1d(t, max_ninf)
                    for it in range(0, len(t) - int(round(icu_stay))):
                        cases_min.append((yV0_int_min(it + icu_stay) - yV0_int_min(it)) * icu_perc)
                        cases_max.append((yV0_int_max(it + icu_stay) - yV0_int_max(it)) * icu_perc)

                    ax.fill_between(t, min_ninf, max_ninf, color=colors[counter], zorder=3, alpha=0.25, linewidth=0)

                    counter += 1

            # Vaccine per group
            for i_fv, frac_vac in enumerate(fractions_vacc_group):

                # Calcualte for group 0
                n_tot = np.sum(fixed_pars[4])
                n_senT3_tot = np.sum(pop)
                n_senT3_inThisT = pop[vaccineGroup[0] - 1]
                n_senVaccine = frac_vac * n_senT3_inThisT
                n_senVaccNotSick = n_senVaccine * effVacsGroup_Sick
                n_toremovedSen = n_senVaccine * effVac
                n_senVaccRemainNotSick = n_senVaccNotSick - n_toremovedSen
                frac_notSick_Sen = n_senVaccRemainNotSick / (n_senT3_tot - n_toremovedSen)

                # Calculate for group 1
                n_NotsenT3_inThisT = fixed_pars[4][vaccineGroup[0] - 1] - n_senT3_inThisT
                n_NotsenVaccine = frac_vac * n_NotsenT3_inThisT
                n_NotsenVaccNotSick = n_NotsenVaccine * effVacsGroup_Sick
                n_toremovedNotSen = n_NotsenVaccine * effVac
                n_NotsenVaccRemainNotSick = n_NotsenVaccNotSick - n_toremovedNotSen
                frac_notSick_NotSen = n_NotsenVaccRemainNotSick / (n_tot - n_toremovedSen - n_toremovedNotSen)

                # Redefine ICU percentage
                icu_perc_use = 0.4 * icu_perc2 * (1 - frac_notSick_NotSen) + icu_perc2 * 0.6 * (1 - frac_notSick_Sen)

                if frac_vac == 0 and ev > 0:
                    continue

                all_ninf = np.zeros((n_chains, len(t)))
                for j in range(0, n_chains):
                    ind = randInds[j]
                    this_params = getPar_list(df_params, ind, i_q)
                    for q in all_quat:
                        if q == all_quat[0]:
                            mdl_cuminf_uncer = predict_VaccineOneGroup(this_params, fixed_pars, frac_vac, t, q,
                                                                       vaccineGroup[i_q], effVac)
                        else:
                            mdl_cuminf_uncer = mdl_cuminf_uncer + predict_VaccineOneGroup(this_params, fixed_pars,
                                                                                          frac_vac, t, q,
                                                                                          vaccineGroup[i_q], effVac)

                    all_ninf[j, :] = mdl_cuminf_uncer

                # Average over boot straps
                mean_ninf = np.median(all_ninf, axis=0)
                min_ninf = np.percentile(all_ninf, 2.275, axis=0)
                max_ninf = np.percentile(all_ninf, 97.275, axis=0)
                # min_ninf = mean_ninf - 2 * np.std(all_ninf, axis=0)
                # max_ninf = mean_ninf + 2 * np.std(all_ninf, axis=0)

                yV2_int = interp1d(t, mean_ninf)
                cases = []
                for it in range(0, len(t) - int(round(icu_stay))):
                    cases.append((yV2_int(it + icu_stay) - yV2_int(it)) * icu_perc_use)

                yV2_int_cases = interp1d(100 * np.array(cases)[:cases.index(max(cases))] / icu_capa,
                                         np.arange(cases.index(max(cases))))
                try:
                    xV2 = yV2_int_cases(50)
                    yV2_int = interp1d(t, mean_ninf)
                    yV2 = yV2_int(xV2)
                except:
                    xV2 = 1000
                    yV2 = 1000

                if frac_vac == 0:
                    ax.plot(np.arange(len(cases)), 100 * np.array(cases) / icu_capa, '-', linewidth=2,
                            color=colors[counter], zorder=12, label='V0 - no vaccine')
                    ax.plot(t, mean_ninf, '-', linewidth=2, color=colors[counter], zorder=12,
                            label='V0 - no vaccine')
                else:
                    if i_fv == 1:
                        if vaccinateALLsenior:
                            ax.plot(t, mean_ninf, '-', linewidth=2, color='red', zorder=12,
                                    label='V3 - ' + f"{frac_vac * 100:.0f}" + '% T' + str(vaccineGroup[0]) + ' ' +
                                          names[0] + '=' + f"{fractions_vaccGroup[0] * 100:.0f}" + '% total')
                            # ax.plot(np.arange(len(cases)), 100 * np.array(cases) / icu_capa, '-', linewidth=2,
                            #         color=colors[counter], zorder=12,
                            #         label='V3 - ' + f"{frac_vac * 100:.0f}" + '% T' + str(vaccineGroup[i_q]) + ' ' +
                            #               names[i_q] + '=' + f"{fractions_vaccGroup[0] * 100:.0f}" + '% total')


                        else:
                            ax.plot(t, mean_ninf, '-', linewidth=2, color='red', zorder=12,
                                    label='V2 - ' + f"{frac_vac * 100:.0f}" + '% T' + str(vaccineGroup[0]) + ' ' +
                                          names[0] + '=' + f"{fractions_vaccGroup[0] * 100:.0f}" + '% total')
                            # ax.plot(np.arange(len(cases)), 100*np.array(cases)/icu_capa, '-', linewidth=2, color=colors[counter], zorder=12,
                            #         label='V2 - '+f"{frac_vac * 100:.0f}" + '% T'+ str(vaccineGroup[i_q]) +' '+ names[i_q] + '=' + f"{fractions_vaccGroup[0]*100:.0f}" + '% total')

                    else:

                        output = {'t': t,
                                  'mean_ninf': mean_ninf,
                                  'min_ninf': min_ninf,
                                  'max_ninf': max_ninf,
                                  'label': 'V2 - ' + f"{fractions_vaccGroup[0] * 100:.0f}" + '% from T' + str(
                                      vaccineGroup[0]) + "\n" + names[0]
                                  }

                        ax.plot(t, mean_ninf, '-', linewidth=2, color='red', zorder=12,
                                label='V2 - ' + f"{fractions_vaccGroup[0] * 100:.0f}" + '% from T' + str(
                                    vaccineGroup[0]) + "\n" + names[0])
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
                    cases_min.append((yV0_int_min(it + icu_stay) - yV0_int_min(it)) * icu_perc_use)
                    cases_max.append((yV0_int_max(it + icu_stay) - yV0_int_max(it)) * icu_perc_use)
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
    plt.xlim([0, len(cases_min)])
    plt.ylabel('Number of Cases', fontsize=14)
    ax.patch.set_facecolor('lightgrey')
    plt.grid(color='white')
    plt.scatter(xV0, yV0, color='black', marker='o', s=50, zorder=20)
    plt.scatter(xV1, yV1, color='darkblue', marker='o', s=50, zorder=20)
    plt.scatter(xV2, yV2, color='red', marker='o', s=50, zorder=20)
    plt.ylim([1, 300000])
    plt.xlim([0, len(cases_min)])
    plt.yscale('log')
    plt.legend(prop={'size': 10}, frameon=True, facecolor='lightgrey', loc='lower right')
    plt.tight_layout()

    os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
    fig1.savefig('vaccineoneGroup_T' + str(vaccineGroup[i_q]) + names[0] + '_mixed_all_3intro_effTransm' + str(
        effVacsGroup[0]) +
                 '_effSick' + str(effVacsGroup_Sick) + '_percVacc' + str(fractions_vaccGroup[0]) + 'LOG.png',
                 dpi=250)
    fig1.savefig('vaccineoneGroup_T' + str(vaccineGroup[i_q]) + names[0] + '_mixed_all_3intro_effTransm' + str(
        effVacsGroup[0]) +
                 '_effSick' + str(effVacsGroup_Sick) + '_percVacc' + str(fractions_vaccGroup[0]) + 'LOG.pdf',
                 format='pdf')
    os.chdir(wd)

    print(xV0)
    print(xV1)
    print(xV2)
    print(yV0)
    print(yV1)
    print(yV2)

    output['xV2'] = xV2
    output['yV2'] = yV2

    return output
def evalVaccination_new_ICUprotection_ICU_EMCEE(i_data_lst, names, df_params, fixed_pars, n_chains=100):
    # general setup
    saveplot = True
    randInds = np.random.randint(df_params.shape[0], size=n_chains)
    output = {}
    all_quat = [1, 2, 3]  # the summarized quaters of Basel to be analysed: choose from 1-9,'all'
    colors = ['black', 'darkblue', 'gold', 'royalblue', 'orange', 'yellow']
    fractions_vaccGroup = [0.23]  # [0.3333333]
    effVacsGroup = [0.9]
    effVacsGroup_Sick = 0.9
    icu_capa = 44
    n_adm = len(all_quat)

    global constantR
    global constantMobility
    global zeroMobility
    n_adm = 3
    if names == ['1-person\nhouseholds']:
        vaccineGroup = [3]
    else:
        vaccineGroup = [1]

    icu_stay = 5.9
    if names[0] == 'Seniority':
        vaccinateALLsenior = True
    else:
        vaccinateALLsenior = False
    if vaccinateALLsenior:
        icu_perc2 = 0.005
    else:
        icu_perc2 = 1 * 0.01

    icu_perc = 1 * 0.01

    # Get populations shared between the two groups
    # Get all population data
    pop_df = pd.read_csv(filenameSocioeconomicData)


    # Get socioeconomic data
    soc_df0 = get_socDF('Seniority')
    soc_df1 = get_socDF(names[0])

    # For each tile get the blocks and population
    blocks0 = soc_df0['BLO_ID'].loc[soc_df0['percentile'] == 3].values

    # Get population of group 0 in the tiles of group 1
    pop = []
    for the_quarter in [1, 2, 3]:
        blocks1 = soc_df1['BLO_ID'].loc[soc_df1['percentile'] == the_quarter].values

        # Overlapping plocks of group 0 and these blocks
        blocks = list(set(blocks0).intersection(blocks1))

        pop.append(pop_df['Population 2017'].loc[pop_df['BlockID'].isin(blocks)].sum())

    fig1 = plt.figure(figsize=(4, 4))
    ax = fig1.add_subplot(1, 1, 1)
    counter = 0
    for i_q in i_data_lst:

        t = list(np.arange(0, 150))
        n_tot = np.sum(fixed_pars[4])
        n_vaccineGroup = fixed_pars[4][vaccineGroup[0] - 1]
        frac_notSick = effVacsGroup_Sick * fractions_vaccGroup[0]
        n_notsick = frac_notSick * n_tot
        f25 = fractions_vaccGroup[0] * n_tot / n_vaccineGroup
        fractions_vacc_group = [f25]
        fractions_vacc_all = [0, fractions_vaccGroup[0]]

        for ev in range(0, len(effVacsGroup)):
            effVac = effVacsGroup[ev]
            n_toremoved = fractions_vaccGroup[0] * n_tot * effVac
            n_notSickremain = n_notsick - n_toremoved
            n_s0 = n_tot - n_toremoved
            perc_notsick = n_notSickremain / n_s0

            # Redefine ICU percentage
            icu_perc_use = icu_perc2 * (1 - perc_notsick)
            icu_perc_rand = icu_perc * (1 - perc_notsick)

            if counter == 0:

                # random distribution of vaccinated subjects
                for i_fv, frac_vac in enumerate(fractions_vacc_all):

                    if frac_vac == 0 and ev > 0:
                        continue

                    all_ninf = np.zeros((n_chains, len(t)))
                    for j in range(0, n_chains):
                        ind = randInds[j]
                        this_params = getPar_list(df_params, ind, i_q)

                        for q in all_quat:
                            if q == all_quat[0]:
                                mdl_cuminf_uncer = predict_Vaccine(this_params, fixed_pars, frac_vac, t, q, effVac)
                            else:
                                mdl_cuminf_uncer = mdl_cuminf_uncer + predict_Vaccine(this_params, fixed_pars, frac_vac,
                                                                                      t, q, effVac)

                        all_ninf[j, :] = mdl_cuminf_uncer

                    # Average over boot straps
                    # mean_ninf = np.median(all_ninf, axis=0)
                    # min_ninf = np.percentile(all_ninf, 2.275, axis=0)
                    # max_ninf = np.percentile(all_ninf, 97.275, axis=0)
                    # min_ninf = mean_ninf - 2 * np.std(all_ninf, axis=0)
                    # max_ninf = mean_ninf + 2 * np.std(all_ninf, axis=0)
                    if frac_vac == 0:

                        all_cases = np.empty((all_ninf.shape[0], int(all_ninf.shape[1] - icu_stay)))
                        for i_chain in range(0, all_ninf.shape[0]):
                            yV0_int = interp1d(t, all_ninf[i_chain, :])
                            cases = []
                            for it in range(0, len(t) - int(round(icu_stay))):
                                cases.append((yV0_int(it + icu_stay) - yV0_int(it)) * icu_perc_rand)
                                # cases.append((all_ninf[i_chain,int(it + icu_stay)] - all_ninf[i_chain,it]) * icu_perc)
                            all_cases[i_chain, :] = cases

                        med_cases = np.percentile(all_cases, 50, axis=0)
                        cases_min = np.percentile(all_cases, 2.275, axis=0)
                        cases_max = np.percentile(all_cases, 97.275, axis=0)

                        yV0_int_cases = interp1d(
                            100 * np.array(med_cases)[:list(med_cases).index(max(med_cases))] / icu_capa,
                            np.arange(list(med_cases).index(max(med_cases))))

                        xV0 = yV0_int_cases(50)

                        ax.plot(np.arange(len(cases)), 100 * np.array(med_cases) / icu_capa, '-', linewidth=2,
                                color=colors[counter], zorder=12,
                                label='V0 - no vaccine')
                    else:

                        all_cases = np.empty((all_ninf.shape[0], int(all_ninf.shape[1] - icu_stay)))
                        for i_chain in range(0, all_ninf.shape[0]):
                            yV1_int = interp1d(t, all_ninf[i_chain, :])
                            cases = []
                            for it in range(0, len(t) - int(round(icu_stay))):
                                cases.append((yV1_int(it + icu_stay) - yV1_int(it)) * icu_perc_rand)
                                # cases.append((all_ninf[i_chain,int(it + icu_stay)] - all_ninf[i_chain,it]) * icu_perc)
                            all_cases[i_chain, :] = cases

                        med_cases = np.percentile(all_cases, 50, axis=0)
                        cases_min = np.percentile(all_cases, 2.275, axis=0)
                        cases_max = np.percentile(all_cases, 97.275, axis=0)

                        yV1_int_cases = interp1d(
                            100 * np.array(med_cases)[:list(med_cases).index(max(med_cases))] / icu_capa,
                            np.arange(list(med_cases).index(max(med_cases))))

                        xV1 = yV1_int_cases(50)

                        ax.plot(np.arange(len(cases)), 100 * med_cases / icu_capa, '-', linewidth=2,
                                color=colors[counter], zorder=12,
                                label='V1 - ' + f"{frac_vac * 100:.0f}" + '% randomly')

                    ax.fill_between(np.arange(len(cases_min)), 100 * cases_min / icu_capa,
                                    100 * cases_max / icu_capa, color=colors[counter], zorder=3,
                                    alpha=0.25, linewidth=0)
                    # ax.plot(np.arange(len(cases_min)), 100 * np.array(cases_min) / icu_capa, '--',
                    #         color=colors[counter], zorder=1, alpha=0.5, linewidth=1)
                    # ax.plot(np.arange(len(cases_min)), 100 * np.array(cases_max) / icu_capa, '--',
                    #         color=colors[counter], zorder=1, alpha=0.5, linewidth=1)

                    counter += 1

            # Vaccine per group
            for i_fv, frac_vac in enumerate(fractions_vacc_group):

                # Calcualte for group 0
                n_tot = np.sum(fixed_pars[4])
                n_senT3_tot = np.sum(pop)
                n_senT3_inThisT = pop[vaccineGroup[0] - 1]
                n_senVaccine = frac_vac * n_senT3_inThisT
                n_senVaccNotSick = n_senVaccine * effVacsGroup_Sick
                n_toremovedSen = n_senVaccine * effVac
                n_senVaccRemainNotSick = n_senVaccNotSick - n_toremovedSen
                frac_notSick_Sen = n_senVaccRemainNotSick / (n_senT3_tot - n_toremovedSen)

                # Calculate for group 1
                n_NotsenT3_inThisT = fixed_pars[4][vaccineGroup[0] - 1] - n_senT3_inThisT
                n_NotsenVaccine = frac_vac * n_NotsenT3_inThisT
                n_NotsenVaccNotSick = n_NotsenVaccine * effVacsGroup_Sick
                n_toremovedNotSen = n_NotsenVaccine * effVac
                n_NotsenVaccRemainNotSick = n_NotsenVaccNotSick - n_toremovedNotSen
                frac_notSick_NotSen = n_NotsenVaccRemainNotSick / (n_tot - n_toremovedSen - n_toremovedNotSen)

                # Redefine ICU percentage
                # icu_perc_use = 0.4 * icu_perc * (1 - frac_notSick_NotSen) + icu_perc * 0.6 * (1 - frac_notSick_Sen)

                if frac_vac == 0 and ev > 0:
                    continue

                all_ninf = np.zeros((n_chains, len(t)))
                for j in range(0, n_chains):
                    ind = randInds[j]
                    this_params = getPar_list(df_params, ind, i_q)
                    for q in all_quat:
                        if q == all_quat[0]:
                            mdl_cuminf_uncer = predict_VaccineOneGroup(this_params, fixed_pars, frac_vac, t, q,
                                                                       vaccineGroup[0], effVac)
                        else:
                            mdl_cuminf_uncer = mdl_cuminf_uncer + predict_VaccineOneGroup(this_params, fixed_pars,
                                                                                          frac_vac, t, q,
                                                                                          vaccineGroup[0], effVac)
                    all_ninf[j, :] = mdl_cuminf_uncer

                all_cases = np.empty((all_ninf.shape[0], int(all_ninf.shape[1] - icu_stay)))
                for i_chain in range(0, all_ninf.shape[0]):
                    yV2_int = interp1d(t, all_ninf[i_chain, :])
                    cases = []
                    for it in range(0, len(t) - int(round(icu_stay))):
                        cases.append((yV2_int(it + icu_stay) - yV2_int(it)) * icu_perc_use)
                        # cases.append((all_ninf[i_chain,int(it + icu_stay)] - all_ninf[i_chain,it]) * icu_perc)
                    all_cases[i_chain, :] = cases

                med_cases = np.percentile(all_cases, 50, axis=0)
                cases_min = np.percentile(all_cases, 2.275, axis=0)
                cases_max = np.percentile(all_cases, 97.275, axis=0)
                yV2_int_cases = interp1d(100 * np.array(med_cases)[:list(med_cases).index(max(med_cases))] / icu_capa,
                                         np.arange(list(med_cases).index(max(med_cases))))

                try:
                    xV2 = yV2_int_cases(50)
                except:
                    xV2 = np.nan

                if frac_vac == 0:
                    ax.plot(np.arange(len(cases)), 100 * np.array(med_cases) / icu_capa, '-', linewidth=2,
                            color=colors[counter], zorder=12, label='V0 - no vaccine')
                    # ax.plot(t, mean_ninf, '-', linewidth=2, color=colors[counter], zorder=12,
                    #         label='V0 - no vaccine')
                else:
                    if i_fv == 1:
                        if vaccinateALLsenior:
                            ax.plot(np.arange(len(cases)), 100 * np.array(med_cases) / icu_capa, '-', linewidth=2,
                                    color='red', zorder=12,
                                    label='V3 - ' + f"{frac_vac * 100:.0f}" + '% T' + str(vaccineGroup[0]) + ' ' +
                                          names[0] + '=' + f"{fractions_vaccGroup[0] * 100:.0f}" + '% total')
                            # ax.plot(np.arange(len(cases)), 100 * np.array(cases) / icu_capa, '-', linewidth=2,
                            #         color=colors[counter], zorder=12,
                            #         label='V3 - ' + f"{frac_vac * 100:.0f}" + '% T' + str(vaccineGroup[i_q]) + ' ' +
                            #               names[i_q] + '=' + f"{fractions_vaccGroup[0] * 100:.0f}" + '% total')


                        else:
                            ax.plot(np.arange(len(cases)), 100 * np.array(med_cases) / icu_capa, '-', linewidth=2,
                                    color='red', zorder=12,
                                    label='V2 - ' + f"{frac_vac * 100:.0f}" + '% T' + str(vaccineGroup[0]) + ' ' +
                                          names[0] + '=' + f"{fractions_vaccGroup[0] * 100:.0f}" + '% total')
                            # ax.plot(np.arange(len(cases)), 100*np.array(cases)/icu_capa, '-', linewidth=2, color=colors[counter], zorder=12,
                            #         label='V2 - '+f"{frac_vac * 100:.0f}" + '% T'+ str(vaccineGroup[i_q]) +' '+ names[i_q] + '=' + f"{fractions_vaccGroup[0]*100:.0f}" + '% total')

                    else:

                        output = {'t': np.arange(len(cases)),
                                  'mean_ninf': 100 * np.array(med_cases) / icu_capa,
                                  'label': 'V2 - ' + f"{fractions_vaccGroup[0] * 100:.0f}" + '% from T' + str(
                                      vaccineGroup[0]) + "\n" + names[0]
                                  }

                        ax.plot(np.arange(len(cases)), 100 * np.array(med_cases) / icu_capa, '-', linewidth=2,
                                color='red', zorder=12,
                                label='V2 - ' + f"{fractions_vaccGroup[0] * 100:.0f}" + '% from T' + str(
                                    vaccineGroup[0]) + "\n" + names[0])
                        # ax.plot(np.arange(len(cases)), 100*np.array(cases)/icu_capa, '-', linewidth=2, color='red', zorder=12,
                        #         label='V2 - '+ f"{fractions_vaccGroup[0]*100:.0f}"+ '% from T'+ str(vaccineGroup[i_q])+ "\n"+names[i_q])

                # Case differences
                # delta = (mean_ninf1 / mean_ninf)
                # list(mean_ninf1 / mean_ninf).index(np.nanmax(mean_ninf1 / mean_ninf))

                # cases_min = []
                # cases_max = []
                # yV0_int_min = interp1d(t, min_ninf)
                # yV0_int_max = interp1d(t, max_ninf)
                # cases = []
                # for it in range(0, len(t) - int(round(icu_stay))):
                #     cases_min.append((yV0_int_min(it + icu_stay) - yV0_int_min(it)) * icu_perc_use)
                #     cases_max.append((yV0_int_max(it + icu_stay) - yV0_int_max(it)) * icu_perc_use)
                # # ax.fill_between(np.arange(len(cases_min)), 100*np.array(cases_min)/icu_capa,
                # #                 100*np.array(cases_max)/icu_capa, color='red', zorder=3,alpha = 0.25, linewidth=0)
                # # ax.plot(np.arange(len(cases_min)), 100 * np.array(cases_min) / icu_capa, '--',
                # #         color='red', zorder=1, alpha=0.5, linewidth=1)
                # # ax.plot(np.arange(len(cases_min)), 100 * np.array(cases_max) / icu_capa, '--',
                # #         color='red', zorder=1, alpha=0.5, linewidth=1)
                output['min_ninf'] = 100 * np.array(cases_min) / icu_capa
                output['max_ninf'] = 100 * np.array(cases_max) / icu_capa

                ax.fill_between(np.arange(len(cases_max)), 100 * np.array(cases_min) / icu_capa,
                                100 * np.array(cases_max) / icu_capa, color='red', zorder=3, alpha=0.25,
                                linewidth=0)
                # ax.plot(np.arange(len(cases_min)), 100 * np.array(cases_min) / icu_capa, '--',
                #         color='red', zorder=1, alpha=0.5, linewidth=1)
                # ax.plot(np.arange(len(cases_min)), 100 * np.array(cases_max) / icu_capa, '--',
                #         color='red', zorder=1, alpha=0.5, linewidth=1)

                counter += 1

    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.xlabel('Time [days]', fontsize=14)
    plt.xlim([0, len(cases_min)])
    plt.ylabel('ICU Occupancy [\%]', fontsize=14)
    ax.patch.set_facecolor('lightgrey')
    plt.grid(color='white')
    plt.scatter(xV0, 50, color='black', marker='o', s=50, zorder=20)
    plt.scatter(xV1, 50, color='darkblue', marker='o', s=50, zorder=20)
    plt.scatter(xV2, 50, color='red', marker='o', s=50, zorder=20)
    plt.ylim([0.1, 1000])
    plt.yscale('log')
    plt.tight_layout()


    os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
    fig1.savefig('vaccineoneGroup_T' + str(vaccineGroup[0]) + names[0] + '_mixed_all_3intro_effTransm' + str(
        effVacsGroup[0]) +
                 '_effSick' + str(effVacsGroup_Sick) + '_percVacc' + str(fractions_vaccGroup[0]) + '_ICU_LOG.png',
                 dpi=250)
    fig1.savefig('vaccineoneGroup_T' + str(vaccineGroup[0]) + names[0] + '_mixed_all_3intro_effTransm' + str(
        effVacsGroup[0]) +
                 '_effSick' + str(effVacsGroup_Sick) + '_percVacc' + str(fractions_vaccGroup[0]) + '_ICU_LOG.pdf',
                 format='pdf')
    os.chdir(wd)

    print(xV0)
    print(xV1)
    print(xV2)

    output['xV2'] = xV2
    output['yV2'] = 50

    return output
def evalVaccination_new_ICUprotection_CombiSenMedInc_EMCEE(i_data_lst, names, names_legend, output, df_params,
                                                           fixed_pars,
                                                           fractions_vaccGroup, fractions_vaccGroup0, n_chains=100):
    global constantR
    global constantMobility
    global zeroMobility

    # general setup
    randInds          = np.random.randint(df_params.shape[0], size=n_chains)
    t                 = list(np.arange(0, 150))
    vaccineGroup      = [3, 1]
    icu_stay          = 5.9
    effVacsGroup      = [0.9]
    effVacsGroup_Sick = 0.9
    icu_capa          = 44
    n_adm             = 3
    all_quat          = [1, 2, 3]  # the summarized quaters of Basel to be analysed: choose from 1-9,'all'
    colors            = ['black', 'darkblue', 'gold', 'royalblue', 'orange', 'yellow']


    if names[0] == 'Seniority':
        vaccinateALLsenior = True
    else:
        vaccinateALLsenior = False

    # ICU percentage used if no prioritization is used for vaccination or no vaccination is used = 50% of ICU pats are >64y
    icu_perc = 1 * 0.01

    # Get populations shared between the two groups

    # Get all population data
    pop_df = pd.read_csv(filenameSocioeconomicData)


    # Get socioeconomic data
    soc_df0 = get_socDF(names[0])
    soc_df1 = get_socDF(names[1])

    # For each tile get the blocks and population
    blocks0 = soc_df0['BLO_ID'].loc[soc_df0['percentile'] == vaccineGroup[0]].values

    # Get population of group 0 in the tiles of group 1
    pop = []
    for the_quarter in range(1, n_adm + 1):
        blocks1 = soc_df1['BLO_ID'].loc[soc_df1['percentile'] == the_quarter].values

        # Overlapping plocks of group 0 and these blocks
        blocks = list(set(blocks0).intersection(blocks1))

        pop.append(pop_df['Population 2017'].loc[pop_df['BlockID'].isin(blocks)].sum())

    fig1 = plt.figure(figsize=(4, 4))
    ax = fig1.add_subplot(1, 1, 1)
    counter = 0

    for i_q in [i_data_lst[1]]:

        n_tot = np.sum(fixed_pars[4])

        # Calcualte for group 0
        perc_senT3 = np.sum(pop) / n_tot
        frac_Sick_Sen = 1 - effVacsGroup_Sick * fractions_vaccGroup0[0]
        frac_notSick_Sen = effVacsGroup_Sick * fractions_vaccGroup0[0]

        ## Recalculate ICU percentage - only if prioritized vaccination happens!
        # icu_perc2 = 0.5*icu_perc + icu_perc*0.5*frac_Sick_Sen

        # Calculate for group 1
        n_vaccineGroup = fixed_pars[4][vaccineGroup[1] - 1]
        frac_notSick = effVacsGroup_Sick * fractions_vaccGroup[0]
        n_notsick = frac_notSick * n_tot

        f25 = fractions_vaccGroup[0] * n_tot / n_vaccineGroup
        fractions_vacc_group = [f25]
        fractions_vacc_all = [0, fractions_vaccGroup[0]]

        for ev in range(0, len(effVacsGroup)):
            effVac = effVacsGroup[ev]
            n_toremoved = fractions_vaccGroup[0] * n_tot * effVac
            n_toremovedSen = perc_senT3 * n_tot * effVac * fractions_vaccGroup0[0]
            n_notsickSen = frac_notSick_Sen * n_tot * perc_senT3
            n_notSickremainSen = n_notsickSen - n_toremovedSen
            n_notSickremain = n_notsick - n_toremoved
            n_s0 = n_tot - n_toremoved - n_toremovedSen
            perc_notsick = n_notSickremain / n_s0
            perc_notsick_random = (n_notSickremain + n_notSickremainSen) / n_s0
            n_moveto_r = n_toremoved + n_toremovedSen

            # Redefine ICU percentage
            icu_perc_use = 0.4 * icu_perc * (1 - perc_notsick) + icu_perc * 0.6 * (1 - frac_notSick_Sen)
            icu_perc_rand = icu_perc * (1 - perc_notsick_random)

            # Define percentages for legend
            frac_vac_random = n_moveto_r / n_tot / effVac

            if counter == 0:

                # random distribution of vaccinated subjects
                for i_fv, frac_vac in enumerate(fractions_vacc_all):

                    if frac_vac == 0 and ev > 0:
                        continue

                    all_ninf = np.zeros((n_chains, len(t)))
                    for j in range(0, n_chains):
                        ind = randInds[j]
                        this_params = getPar_list(df_params, ind, i_q)
                        mdl_cuminf_uncer = predict_Vaccine_Combi(this_params, fixed_pars, frac_vac, t, effVac,
                                                                 n_moveto_r)
                        all_ninf[j, :] = mdl_cuminf_uncer

                    # Average over boot straps
                    mean_ninf = np.median(all_ninf, axis=0)
                    min_ninf = np.percentile(all_ninf, 2.275, axis=0)
                    max_ninf = np.percentile(all_ninf, 97.275, axis=0)
                    # min_ninf = mean_ninf - 2 * np.std(all_ninf, axis=0)
                    # max_ninf = mean_ninf + 2 * np.std(all_ninf, axis=0)
                    if frac_vac == 0:

                        yV0_int = interp1d(t, mean_ninf)
                        cases = []
                        for it in range(0, len(t) - int(round(icu_stay))):
                            cases.append((yV0_int(it + icu_stay) - yV0_int(it)) * icu_perc_rand)

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
                            cases.append((yV1_int(it + icu_stay) - yV1_int(it)) * icu_perc_rand)

                        yV1_int_cases = interp1d(100 * np.array(cases)[:cases.index(max(cases))] / icu_capa,
                                                 np.arange(cases.index(max(cases))))

                        try:
                            xV1 = yV1_int_cases(50)
                            yV1_int = interp1d(t, mean_ninf)
                            yV1 = yV1_int(xV1)
                        except:
                            xV1 = 1000
                            yV1 = np.nan

                        # ax.plot(np.arange(len(cases)), 100*np.array(cases)/icu_capa, '-', linewidth=2, color=colors[counter], zorder=12,
                        #         label='V1 - ' + f"{frac_vac * 100:.0f}" + '% randomly')  # with '+ f"{effVac*100:.0f}"+'% efficacy')
                        ax.plot(t, mean_ninf, '-', linewidth=2, color=colors[counter], zorder=12,
                                label='V1 - ' + f"{frac_vac_random * 100:.0f}" + '% randomly')

                    # cases_min = []
                    # cases_max = []
                    # yV0_int_min = interp1d(t, min_ninf)
                    # yV0_int_max = interp1d(t, max_ninf)
                    # for it in range(0, len(t) - int(round(icu_stay))):
                    #     cases_min.append((yV0_int_min(it + icu_stay) - yV0_int_min(it)) * icu_perc)
                    #     cases_max.append((yV0_int_max(it + icu_stay) - yV0_int_max(it)) * icu_perc)

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

                all_ninf = np.zeros((n_chains, len(t)))
                for j in range(0, n_chains):
                    ind = randInds[j]
                    this_params = getPar_list(df_params, ind, i_q)

                    mdl_cuminf_uncer = predict_VaccineOneGroup_Combi(this_params, fixed_pars, frac_vac, t,
                                                                     vaccineGroup[1], effVac, n_toremoved,
                                                                     n_toremovedSen, pop)

                    all_ninf[j, :] = mdl_cuminf_uncer

                # Average over boot straps
                mean_ninf = np.median(all_ninf, axis=0)
                min_ninf = np.percentile(all_ninf, 2.275, axis=0)
                max_ninf = np.percentile(all_ninf, 97.275, axis=0)
                # min_ninf = mean_ninf - 2 * np.std(all_ninf, axis=0)
                # max_ninf = mean_ninf + 2 * np.std(all_ninf, axis=0)

                yV2_int = interp1d(t, mean_ninf)
                cases = []
                for it in range(0, len(t) - int(round(icu_stay))):
                    cases.append((yV2_int(it + icu_stay) - yV2_int(it)) * icu_perc_use)

                yV2_int_cases = interp1d(100 * np.array(cases)[:cases.index(max(cases))] / icu_capa,
                                         np.arange(cases.index(max(cases))))

                try:
                    xV2 = yV2_int_cases(50)
                    yV2_int = interp1d(t, mean_ninf)
                    yV2 = yV2_int(xV2)
                except:
                    xV2 = 1000
                    yV2 = 1000

                if frac_vac == 0:
                    ax.plot(np.arange(len(cases)), 100 * np.array(cases) / icu_capa, '-', linewidth=2,
                            color=colors[counter], zorder=12, label='V0 - no vaccine')
                    ax.plot(t, mean_ninf, '-', linewidth=2, color=colors[counter], zorder=12,
                            label='V0 - no vaccine')
                else:
                    if i_fv == 1:
                        if vaccinateALLsenior:
                            ax.plot(t, mean_ninf, '-', linewidth=2, color='red', zorder=12,
                                    label='V3 - ' + f"{0.23 * 100:.0f}" + '%in T' + str(vaccineGroup[0]) + ' ' +
                                          names_legend[0] + '+' f"{0.19 * 100:.0f}" + '%in T' + str(
                                        vaccineGroup[1]) + ' ' +
                                          names_legend[1])
                            # ax.plot(np.arange(len(cases)), 100 * np.array(cases) / icu_capa, '-', linewidth=2,
                            #         color=colors[counter], zorder=12,
                            #         label='V3 - ' + f"{frac_vac * 100:.0f}" + '% T' + str(vaccineGroup[i_q]) + ' ' +
                            #               names[i_q] + '=' + f"{fractions_vaccGroup[0] * 100:.0f}" + '% total')


                        else:
                            ax.plot(t, mean_ninf, '-', linewidth=2, color='red', zorder=12,
                                    label='V3 - ' + f"{0.23 * 100:.0f}" + '%in T' + str(vaccineGroup[0]) + ' ' +
                                          names_legend[0] + '+' f"{0.19 * 100:.0f}" + '%in T' + str(
                                        vaccineGroup[1]) + ' ' +
                                          names_legend[1])
                            # ax.plot(np.arange(len(cases)), 100*np.array(cases)/icu_capa, '-', linewidth=2, color=colors[counter], zorder=12,
                            #         label='V2 - '+f"{frac_vac * 100:.0f}" + '% T'+ str(vaccineGroup[i_q]) +' '+ names[i_q] + '=' + f"{fractions_vaccGroup[0]*100:.0f}" + '% total')

                    else:
                        if output is not None:
                            ax.plot(output['t'], output['mean_ninf'], '-', linewidth=2, color='darkorange', zorder=12,
                                    label=output['label'])
                            ax.fill_between(output['t'], output['min_ninf'], output['max_ninf'], color='darkorange',
                                            zorder=3, alpha=0.25, linewidth=0)
                            plt.scatter(output['xV2'], output['yV2'], color='darkorange', marker='o', s=50, zorder=20)

                        if fractions_vaccGroup[0] < 0.05:
                            ax.plot(t, mean_ninf, '-', linewidth=2, color='red', zorder=12,
                                    label='V3 - ' + f"{fractions_vaccGroup0[0] * 0.23 * 100:.0f}" + '% from T' + str(
                                        vaccineGroup[0]) + "\n" + names_legend[0])
                            # ax.plot(np.arange(len(cases)), 100*np.array(cases)/icu_capa, '-', linewidth=2, color='red', zorder=12,
                            #         label='V2 - '+ f"{fractions_vaccGroup[0]*100:.0f}"+ '% from T'+ str(vaccineGroup[i_q])+ "\n"+names[i_q])
                        else:
                            ax.plot(t, mean_ninf, '-', linewidth=2, color='red', zorder=12,
                                    label='V4 - ' + f"{fractions_vaccGroup[0] * 100:.1f}" + '% from T' + str(
                                        vaccineGroup[1]) + '\n' +
                                          names_legend[
                                              1] + "\n" + '+' f"{fractions_vaccGroup0[0] * 0.23 * 100:.1f}" + '% T'
                                          + str(vaccineGroup[0]) + ' ' + names_legend[0])

                            # ax.plot(np.arange(len(cases)), 100*np.array(cases)/icu_capa, '-', linewidth=2, color='red', zorder=12,
                            #         label='V2 - '+ f"{fractions_vaccGroup[0]*100:.0f}"+ '% from T'+ str(vaccineGroup[i_q])+ "\n"+names[i_q])

                # Case differences
                delta = (mean_ninf1 / mean_ninf)
                list(mean_ninf1 / mean_ninf).index(np.nanmax(mean_ninf1 / mean_ninf))

                # cases_min = []
                # cases_max = []
                # yV0_int_min = interp1d(t, min_ninf)
                # yV0_int_max = interp1d(t, max_ninf)
                # cases = []
                # for it in range(0, len(t) - int(round(icu_stay))):
                #     cases_min.append((yV0_int_min(it + icu_stay) - yV0_int_min(it)) * icu_perc_use)
                #     cases_max.append((yV0_int_max(it + icu_stay) - yV0_int_max(it)) * icu_perc_use)
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
    plt.yscale('linear')
    plt.ylabel('Number of Cases', fontsize=14)
    ax.patch.set_facecolor('lightgrey')
    plt.grid(color='white')
    plt.scatter(xV0, yV0, color='black', marker='o', s=50, zorder=20)
    plt.scatter(xV1, yV1, color='darkblue', marker='o', s=50, zorder=20)
    plt.scatter(xV2, yV2, color='red', marker='o', s=50, zorder=20)
    plt.ylim([1, 300000])
    plt.xlim([0., len(cases)])
    plt.yscale('log')
    plt.legend(prop={'size': 10}, frameon=True, facecolor='lightgrey', loc='lower right')
    plt.tight_layout()

    if output is not None:
        suffix = '4plot'
    else:
        suffix = ''
    os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
    fig1.savefig(
        'vaccineoneGroup_T' + str(vaccineGroup[0]) + str(names) + '_mixed_all_COMBI_3intro_effTransm' + str(
            effVacsGroup[0]) + '_effSick' + str(effVacsGroup_Sick) + '_percVacc' + str(
            fractions_vaccGroup[0]) + '_LOG_' + suffix + '.png', dpi=250)
    fig1.savefig(
        'vaccineoneGroup_T' + str(vaccineGroup[0]) + str(names) + '_mixed_all_COMBI_3intro_effTransm' + str(
            effVacsGroup[0]) + '_effSick' + str(effVacsGroup_Sick) + '_percVacc' + str(
            fractions_vaccGroup[0]) + '_LOG_' + suffix + '.pdf', format='pdf')
    os.chdir(wd)

    print(xV0)
    print(xV1)
    print(xV2)
    print(yV0)
    print(yV1)
    print(yV2)

    return 0
def evalVaccination_new_ICUprotection_CombiSenMedInc_ICU_EMCEE(i_data_lst, names, names_legend, output, df_params,
                                                               fixed_pars,
                                                               fractions_vaccGroup, fractions_vaccGroup0, n_chains=100):
    global constantR
    global constantMobility
    global zeroMobility


    # general setup
    randInds          = np.random.randint(df_params.shape[0], size=n_chains)
    vaccineGroup      = [3, 1]
    icu_stay          = 5.9
    effVacsGroup      = [0.9]
    effVacsGroup_Sick = 0.9
    icu_capa          = 44
    t                 = np.arange(150)
    n_adm = 3
    all_quat     = [1, 2, 3]  # the summarized quaters of Basel to be analysed: choose from 1-9,'all'
    colors       = ['black', 'darkblue', 'gold', 'royalblue', 'orange', 'yellow']


    if names[0] == 'Seniority':
        vaccinateALLsenior = True
    else:
        vaccinateALLsenior = False

    # ICU percentage used if no prioritization is used for vaccination or no vaccination is used = 50% of ICU pats are >64y
    icu_perc = 1 * 0.01

    # Get populations shared between the two groups
    # Get all population data
    pop_df = pd.read_csv(filenameSocioeconomicData)


    # Get socioeconomic data
    soc_df0 = get_socDF(names[0])
    soc_df1 = get_socDF(names[1])

    # For each tile get the blocks and population
    blocks0 = soc_df0['BLO_ID'].loc[soc_df0['percentile'] == vaccineGroup[0]].values

    # Get population of group 0 in the tiles of group 1
    pop = []
    for the_quarter in range(1, n_adm + 1):
        blocks1 = soc_df1['BLO_ID'].loc[soc_df1['percentile'] == the_quarter].values

        # Overlapping plocks of group 0 and these blocks
        blocks = list(set(blocks0).intersection(blocks1))

        pop.append(pop_df['Population 2017'].loc[pop_df['BlockID'].isin(blocks)].sum())

    fig1 = plt.figure(figsize=(4, 4))
    ax = fig1.add_subplot(1, 1, 1)
    counter = 0


    for i_q in [i_data_lst[1]]:

        # load data for this run
        # folder = os.path.join(wd, 'Results', run_ID, 'original')
        # files = os.listdir(folder)
        # for i, f in enumerate(files):
        #     if f[0] == str(1) and f[-7:] == 'trn.csv':
        #         df_trn = pd.read_csv(os.path.join(folder, f))
        #         t_trn = df_trn['timestamp'].values
        #     elif f[0] == str(1) and f[-7:] == 'tst.csv':
        #         df_tst = pd.read_csv(os.path.join(folder, f))
        #         t_tst = df_tst['timestamp'].values
        # t = list(np.arange(0, 150))
        #
        # folder = os.path.join(wd, 'Results', run_ID, 'parameters')
        # files = os.listdir(folder)
        # for i, f in enumerate(files):
        #     if f[-9:] == 'itted.pkl':
        #         infile = open(os.path.join(folder, f), 'rb')
        #         result = pickle.load(infile)
        #         infile.close()
        #
        #     elif f[-9:] == 'fixed.pkl':
        #         infile = open(os.path.join(folder, f), 'rb')
        #         fixed_pars = pickle.load(infile)
        #         infile.close()

        n_tot = np.sum(fixed_pars[4])

        # Calcualte for group 0
        perc_senT3 = np.sum(pop) / n_tot
        frac_Sick_Sen = 1 - effVacsGroup_Sick * fractions_vaccGroup0[0]
        frac_notSick_Sen = effVacsGroup_Sick * fractions_vaccGroup0[0]

        ## Recalculate ICU percentage - only if prioritized vaccination happens!
        # icu_perc2 = 0.5*icu_perc + icu_perc*0.5*frac_Sick_Sen

        # Calculate for group 1
        n_vaccineGroup = fixed_pars[4][vaccineGroup[1] - 1]
        frac_notSick = effVacsGroup_Sick * fractions_vaccGroup[0]
        n_notsick = frac_notSick * n_tot

        f25 = fractions_vaccGroup[0] * n_tot / n_vaccineGroup
        fractions_vacc_group = [f25]
        fractions_vacc_all = [0, fractions_vaccGroup[0]]

        for ev in range(0, len(effVacsGroup)):
            effVac = effVacsGroup[ev]
            n_toremoved = fractions_vaccGroup[0] * n_tot * effVac
            n_toremovedSen = perc_senT3 * n_tot * effVac * fractions_vaccGroup0[0]
            n_notsickSen = frac_notSick_Sen * n_tot * perc_senT3
            n_notSickremainSen = n_notsickSen - n_toremovedSen
            n_notSickremain = n_notsick - n_toremoved
            n_s0 = n_tot - n_toremoved - n_toremovedSen
            perc_notsick = n_notSickremain / n_s0
            perc_notsick_random = (n_notSickremain + n_notSickremainSen) / n_s0
            n_moveto_r = n_toremoved + n_toremovedSen

            # Redefine ICU percentage
            icu_perc_use = 0.4 * icu_perc * (1 - perc_notsick) + icu_perc * 0.6 * (1 - frac_notSick_Sen)
            icu_perc_rand = icu_perc * (1 - perc_notsick_random)

            # Define percentages for legend
            frac_vac_random = n_moveto_r / n_tot / effVac

            if counter == 0:

                # random distribution of vaccinated subjects
                for i_fv, frac_vac in enumerate(fractions_vacc_all):

                    if frac_vac == 0 and ev > 0:
                        continue

                    all_ninf = np.zeros((n_chains, len(t)))
                    for j in range(0, n_chains):
                        ind = randInds[j]
                        this_params = getPar_list(df_params, ind, i_q)
                        mdl_cuminf_uncer = predict_Vaccine_Combi(this_params, fixed_pars, frac_vac, t, effVac,
                                                                 n_moveto_r)

                        all_ninf[j, :] = mdl_cuminf_uncer

                    if frac_vac == 0:

                        all_cases = np.empty((all_ninf.shape[0], int(all_ninf.shape[1] - icu_stay)))
                        for i_chain in range(0, all_ninf.shape[0]):
                            yV0_int = interp1d(t, all_ninf[i_chain, :])
                            cases = []
                            for it in range(0, len(t) - int(round(icu_stay))):
                                cases.append((yV0_int(it + icu_stay) - yV0_int(it)) * icu_perc_rand)
                                # cases.append((all_ninf[i_chain,int(it + icu_stay)] - all_ninf[i_chain,it]) * icu_perc)
                            all_cases[i_chain, :] = cases

                        med_cases = np.percentile(all_cases, 50, axis=0)
                        cases_min = np.percentile(all_cases, 2.275, axis=0)
                        cases_max = np.percentile(all_cases, 97.275, axis=0)
                        yV0_int_cases = interp1d(
                            100 * np.array(med_cases)[:list(med_cases).index(max(med_cases))] / icu_capa,
                            np.arange(list(med_cases).index(max(med_cases))))
                        xV0 = yV0_int_cases(50)

                        # ax.plot(np.arange(len(cases)), 100*np.array(cases)/icu_capa, '-', linewidth=2,
                        # color=colors[counter], zorder=12,label='V0 - no vaccine')
                        ax.plot(np.arange(len(cases)), 100 * np.array(med_cases) / icu_capa, '-', linewidth=2,
                                color=colors[counter], zorder=12,
                                label='V0 - no vaccine')
                    else:

                        all_cases = np.empty((all_ninf.shape[0], int(all_ninf.shape[1] - icu_stay)))
                        for i_chain in range(0, all_ninf.shape[0]):
                            yV1_int = interp1d(t, all_ninf[i_chain, :])
                            cases = []
                            for it in range(0, len(t) - int(round(icu_stay))):
                                cases.append((yV1_int(it + icu_stay) - yV1_int(it)) * icu_perc_rand)
                                # cases.append((all_ninf[i_chain,int(it + icu_stay)] - all_ninf[i_chain,it]) * icu_perc)
                            all_cases[i_chain, :] = cases

                        med_cases = np.percentile(all_cases, 50, axis=0)
                        cases_min = np.percentile(all_cases, 2.275, axis=0)
                        cases_max = np.percentile(all_cases, 97.275, axis=0)
                        yV1_int_cases = interp1d(
                            100 * np.array(med_cases)[:list(med_cases).index(max(med_cases))] / icu_capa,
                            np.arange(list(med_cases).index(max(med_cases))))
                        xV1 = yV1_int_cases(50)

                        # ax.plot(np.arange(len(cases)), 100*np.array(cases)/icu_capa, '-', linewidth=2, color=colors[counter], zorder=12,
                        #         label='V1 - ' + f"{frac_vac * 100:.0f}" + '% randomly')  # with '+ f"{effVac*100:.0f}"+'% efficacy')
                        ax.plot(np.arange(len(cases)), 100 * np.array(med_cases) / icu_capa, '-', linewidth=2,
                                color=colors[counter], zorder=12,
                                label='V1 - ' + f"{frac_vac_random * 100:.0f}" + '% randomly')

                    ax.fill_between(np.arange(len(cases_min)), 100 * np.array(cases_min) / icu_capa,
                                    100 * np.array(cases_max) / icu_capa, color=colors[counter], zorder=3, alpha=0.25,
                                    linewidth=0)
                    # ax.plot(np.arange(len(cases_min)), 100 * np.array(cases_min) / icu_capa, '--',
                    #         color=colors[counter], zorder=1, alpha=0.5, linewidth=1)
                    # ax.plot(np.arange(len(cases_min)), 100 * np.array(cases_max) / icu_capa, '--',
                    #         color=colors[counter], zorder=1, alpha=0.5, linewidth=1)
                    counter += 1

            # Vaccine per group
            for i_fv, frac_vac in enumerate(fractions_vacc_group):

                if frac_vac == 0 and ev > 0:
                    continue

                all_ninf = np.zeros((n_chains, len(t)))
                for j in range(0, n_chains):
                    ind = randInds[j]
                    this_params = getPar_list(df_params, ind, i_q)

                    mdl_cuminf_uncer = predict_VaccineOneGroup_Combi(this_params, fixed_pars, frac_vac, t,
                                                                     vaccineGroup[1], effVac, n_toremoved,
                                                                     n_toremovedSen, pop)

                    all_ninf[j, :] = mdl_cuminf_uncer

                all_cases = np.empty((all_ninf.shape[0], int(all_ninf.shape[1] - icu_stay)))
                for i_chain in range(0, all_ninf.shape[0]):
                    yV2_int = interp1d(t, all_ninf[i_chain, :])
                    cases = []
                    for it in range(0, len(t) - int(round(icu_stay))):
                        cases.append((yV2_int(it + icu_stay) - yV2_int(it)) * icu_perc_use)
                        # cases.append((all_ninf[i_chain,int(it + icu_stay)] - all_ninf[i_chain,it]) * icu_perc)
                    all_cases[i_chain, :] = cases

                med_cases = np.percentile(all_cases, 50, axis=0)
                cases_min = np.percentile(all_cases, 2.275, axis=0)
                cases_max = np.percentile(all_cases, 97.275, axis=0)

                yV2_int_cases = interp1d(100 * np.array(med_cases)[:list(med_cases).index(max(med_cases))] / icu_capa,
                                         np.arange(list(med_cases).index(max(med_cases))))

                try:
                    xV2 = yV2_int_cases(50)
                except:
                    xV2 = np.nan

                if frac_vac == 0:
                    ax.plot(np.arange(len(cases)), 100 * np.array(med_cases) / icu_capa, '-', linewidth=2,
                            color=colors[counter], zorder=12, label='V0 - no vaccine')
                else:
                    if i_fv == 1:
                        if vaccinateALLsenior:
                            ax.plot(np.arange(len(cases)), 100 * np.array(med_cases) / icu_capa, '-', linewidth=2,
                                    color='red', zorder=12,
                                    label='V3 - ' + f"{0.23 * 100:.0f}" + '%in T' + str(vaccineGroup[0]) + ' ' +
                                          names_legend[0] + '+' f"{0.19 * 100:.0f}" + '%in T' + str(
                                        vaccineGroup[1]) + ' ' +
                                          names_legend[1])
                            # ax.plot(np.arange(len(cases)), 100 * np.array(cases) / icu_capa, '-', linewidth=2,
                            #         color=colors[counter], zorder=12,
                            #         label='V3 - ' + f"{frac_vac * 100:.0f}" + '% T' + str(vaccineGroup[i_q]) + ' ' +
                            #               names[i_q] + '=' + f"{fractions_vaccGroup[0] * 100:.0f}" + '% total')


                        else:
                            ax.plot(np.arange(len(cases)), 100 * np.array(med_cases) / icu_capa, '-', linewidth=2,
                                    color='red', zorder=12,
                                    label='V3 - ' + f"{0.23 * 100:.0f}" + '%in T' + str(vaccineGroup[0]) + ' ' +
                                          names_legend[0] + '+' f"{0.19 * 100:.0f}" + '%in T' + str(
                                        vaccineGroup[1]) + ' ' +
                                          names_legend[1])
                            # ax.plot(np.arange(len(cases)), 100*np.array(cases)/icu_capa, '-', linewidth=2, color=colors[counter], zorder=12,
                            #         label='V2 - '+f"{frac_vac * 100:.0f}" + '% T'+ str(vaccineGroup[i_q]) +' '+ names[i_q] + '=' + f"{fractions_vaccGroup[0]*100:.0f}" + '% total')

                    else:
                        if output is not None:
                            ax.plot(output['t'], output['mean_ninf'], '-', linewidth=2, color='darkorange', zorder=12,
                                    label=output['label'])
                            ax.fill_between(output['t'], output['min_ninf'], output['max_ninf'], color='darkorange',
                                            zorder=3, alpha=0.25, linewidth=0)
                            # ax.plot(np.arange(len(output['min_ninf'])), output['min_ninf'], '--',
                            #         color='darkorange', zorder=1, alpha=0.5, linewidth=1)
                            # ax.plot(np.arange(len(output['min_ninf'])),  output['max_ninf'], '--',
                            #         color='darkorange', zorder=1, alpha=0.5, linewidth=1)
                            plt.scatter(output['xV2'], output['yV2'], color='darkorange', marker='o', s=50, zorder=20)

                        if fractions_vaccGroup[0] < 0.05:
                            ax.plot(np.arange(len(cases)), 100 * np.array(med_cases) / icu_capa, '-', linewidth=2,
                                    color='red', zorder=12,
                                    label='V3 - ' + f"{fractions_vaccGroup0[0] * 0.23 * 100:.0f}" + '% from T' + str(
                                        vaccineGroup[0]) + "\n" + names_legend[0])
                            # ax.plot(np.arange(len(cases)), 100*np.array(cases)/icu_capa, '-', linewidth=2, color='red', zorder=12,
                            #         label='V2 - '+ f"{fractions_vaccGroup[0]*100:.0f}"+ '% from T'+ str(vaccineGroup[i_q])+ "\n"+names[i_q])
                        else:
                            ax.plot(np.arange(len(cases)), 100 * np.array(med_cases) / icu_capa, '-', linewidth=2,
                                    color='red', zorder=12,
                                    label='V4 - ' + f"{fractions_vaccGroup0[0] * 0.23 * 100:.1f}" + '% from T' + str(
                                        vaccineGroup[0]) + '\n' +
                                          names_legend[
                                              0] + "\n" + '+' f"{fractions_vaccGroup[0] * 100:.1f}" + '% T' + str(
                                        vaccineGroup[1]) +
                                          ' ' + names_legend[1])
                            # ax.plot(np.arange(len(cases)), 100*np.array(cases)/icu_capa, '-', linewidth=2, color='red', zorder=12,
                            #         label='V2 - '+ f"{fractions_vaccGroup[0]*100:.0f}"+ '% from T'+ str(vaccineGroup[i_q])+ "\n"+names[i_q])

                # Case differences
                ax.fill_between(np.arange(len(cases_min)), 100 * np.array(cases_min) / icu_capa,
                                100 * np.array(cases_max) / icu_capa, color='red', zorder=3, alpha=0.25, linewidth=0)
                # ax.plot(np.arange(len(cases_min)), 100*np.array(cases_min)/icu_capa, '--',
                #         color='red', zorder=1, alpha=0.5, linewidth=1)
                # ax.plot(np.arange(len(cases_min)), 100*np.array(cases_max)/icu_capa, '--',
                #         color='red', zorder=1, alpha=0.5, linewidth=1)
                counter += 1

    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.xlabel('Time [days]', fontsize=14)
    plt.legend(prop={'size': 10}, frameon=True, facecolor='lightgrey', loc='upper left')
    plt.xlim([0, len(cases_min)])
    plt.ylabel('ICU Occupancy', fontsize=14)
    ax.patch.set_facecolor('lightgrey')
    plt.grid(color='white')
    plt.scatter(xV0, 50, color='black', marker='o', s=50, zorder=20)
    plt.scatter(xV1, 50, color='darkblue', marker='o', s=50, zorder=20)
    plt.scatter(xV2, 50, color='red', marker='o', s=50, zorder=20)
    plt.ylim([0.01, 1000])
    plt.yscale('log')
    plt.legend(prop={'size': 10}, frameon=True, facecolor='lightgrey')
    plt.tight_layout()

    if output is not None:
        suffix = '4plot'
    else:
        suffix = ''


    os.chdir(os.path.join(wd, 'Results', run_ID, 'figures'))
    fig1.savefig(
        'vaccineoneGroup_T' + str(vaccineGroup[0]) + str(names) + '_mixed_all_COMBI_3intro_effTransm' + str(
            effVacsGroup[0]) + '_effSick' + str(effVacsGroup_Sick) + '_percVacc' + str(
            fractions_vaccGroup[0]) + suffix + '_LOG_ICU.png', dpi=250)
    fig1.savefig(
        'vaccineoneGroup_T' + str(vaccineGroup[0]) + str(names) + '_mixed_all_COMBI_3intro_effTransm' + str(
            effVacsGroup[0]) + '_effSick' + str(effVacsGroup_Sick) + '_percVacc' + str(
            fractions_vaccGroup[0]) + suffix + '_LOG_ICU.pdf', format='pdf')
    os.chdir(wd)

    print(xV0)
    print(xV1)
    print(xV2)

    return 0
def predict_VaccineOneGroup(parameters_fit, fixed_pars_in, frac_vaccinated_in, t, q, vaccineGroup, effVac):
    n_adm = fixed_pars_in[0]
    global constantMobility
    global constantR
    global fixedSocial

    fixedSocial = 0.5 * np.ones((n_adm,))
    fixedSocial[vaccineGroup - 1] = ((1 - frac_vaccinated_in) * 0.5 + frac_vaccinated_in * 0.75 * (1 - effVac)) / (
                1 - frac_vaccinated_in + frac_vaccinated_in * (1 - effVac))
    # (1-frac_vaccinated_in)*0.5+frac_vaccinated_in*0.75#((1-frac_vaccinated_in)*0.5+frac_vaccinated_in*(1-effVac))/(1-frac_vaccinated_in+frac_vaccinated_in*(1-effVac))

    frac_vaccinated = frac_vaccinated_in * effVac

    # Change number of susceptibles and recovered
    n_sus = [[] for i in range(0, n_adm)]
    n_rec = fixed_pars_in[4][vaccineGroup - 1] * frac_vaccinated
    for i in range(0, n_adm):
        if i == vaccineGroup - 1:
            n_sus[i] = fixed_pars_in[4][i] * (1 - frac_vaccinated)
        else:
            n_sus[i] = fixed_pars_in[4][i]

            # intialize a single exposed case in tile 1

    pars_vac90 = parameters_fit.copy()
    pars_vac90[n_adm * 0:n_adm * 5] = 0
    pars_vac90[0:n_adm] = 1
    pars_vac90[n_adm * 4 + vaccineGroup - 1] = n_rec

    # parameters
    fixed_pars_vac90 = fixed_pars_in[:4].copy()
    fixed_pars_vac90.append(n_sus)

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
def predict_VaccineOneGroup_Combi(parameters_fit, fixed_pars_in, frac_vaccinated_in, t, vaccineGroup, effVac, n_toR_1,
                                  n_toR_0, pop):
    n_adm = fixed_pars_in[0]
    global constantMobility
    global constantR
    global fixedSocial

    n_adm = fixed_pars_in[0]
    n_tot = np.sum(fixed_pars_in[4])
    frac_vaccinated_in = (n_toR_0 + n_toR_1) / n_tot / effVac

    # Change number of susceptibles and recovered
    n_sus = [fixed_pars_in[4][i] for i in range(0, n_adm)]
    n_rec = [0 for i in range(0, n_adm)]

    # Account for group 1
    for i in range(0, n_adm):
        if i == vaccineGroup - 1:
            n_sus[i] += - n_toR_1
            n_rec[i] += n_toR_1

    # Account for group 0
    n_move = n_toR_0 * np.array(pop) / np.sum(pop)
    n_sus += - n_move
    n_rec += n_move

    # Account for social behavious changes
    n_remain_vacc = (1 - effVac) * n_rec / effVac
    eff_fracVac = n_rec / effVac / (n_sus + n_rec)
    fixedSocial = ((n_sus - n_remain_vacc) * 0.5 + n_remain_vacc * 0.75) / (n_sus)


    # intialize a single exposed case in each tertile
    pars_vac90 = parameters_fit.copy()
    pars_vac90[n_adm * 0:n_adm * 5] = 0
    pars_vac90[0:n_adm] = 1
    pars_vac90[n_adm * 4:n_adm * 4 + 3] = n_rec

    # parameters
    fixed_pars_vac90 = fixed_pars_in[:4].copy()
    fixed_pars_vac90.append(n_sus)


    # Predict
    constantMobility = True
    constantR = True
    fit_vaccine = solution_SEUI(t, pars_vac90, fixed_pars_vac90).T
    constantMobility = False
    constantR = False
    fixedSocial = 1

    n_cases = 0
    for q in range(0, n_adm):
        n_U = fit_vaccine[2 * n_adm + q - 1]
        n_I = fit_vaccine[3 * n_adm + q - 1]
        n_Ui = fit_vaccine[4 * n_adm + q - 1]
        n_RUi = fit_vaccine[5 * n_adm + q - 1] - fit_vaccine[5 * n_adm + q - 1][0]
        n_cases += n_I + n_Ui + n_RUi

    return n_cases
def predict_Vaccine_Combi(parameters_fit, fixed_pars, frac_vaccinated_in, t, effVac, n_toR):
    n_adm = fixed_pars[0]
    global constantMobility
    global constantR
    global fixedSocial

    n_tot = np.sum(fixed_pars[4])
    if frac_vaccinated_in > 0:
        frac_vaccinated = n_toR / n_tot
        frac_vaccinated_in = n_toR / n_tot / effVac
    else:
        frac_vaccinated = 0

    fixedSocial = ((1 - frac_vaccinated_in) * 0.5 + frac_vaccinated_in * 0.75 * (1 - effVac)) / (
                1 - frac_vaccinated_in + frac_vaccinated_in * (1 - effVac))
    # ((1-frac_vaccinated_in)*0.5+frac_vaccinated_in*0.75)#*(1-effVac))#/(1-frac_vaccinated_in+frac_vaccinated_in*(1-effVac))

    # intialize a single exposed case in tile 1
    pars_vac90 = parameters_fit.copy()
    pars_vac90[n_adm * 0:n_adm * 5] = 0
    pars_vac90[0:n_adm] = 1
    pars_vac90[n_adm * 4:n_adm * 5] = np.array(fixed_pars[4]) * frac_vaccinated

    # parameters
    fixed_pars_vac90 = fixed_pars[0:4].copy()
    fixed_pars_vac90.append(list(np.array(fixed_pars[4]) * (1 - frac_vaccinated)))

    # Predict
    constantMobility = True
    constantR = True
    fit_vaccine = solution_SEUI(t, pars_vac90, fixed_pars_vac90).T
    constantMobility = False
    constantR = False
    fixedSocial = 1

    n_cases = 0
    for q in range(0, n_adm):
        n_U = fit_vaccine[2 * n_adm + q - 1]
        n_I = fit_vaccine[3 * n_adm + q - 1]
        n_Ui = fit_vaccine[4 * n_adm + q - 1]
        n_RUi = fit_vaccine[5 * n_adm + q - 1] - fit_vaccine[5 * n_adm + q - 1][0]
        n_cases += n_I + n_Ui + n_RUi

    return n_cases
def predict_Vaccine(parameters_fit, fixed_pars, frac_vaccinated_in, t, q, effVac):
    n_adm = fixed_pars[0]
    global constantMobility
    global constantR
    global fixedSocial

    fixedSocial = ((1 - frac_vaccinated_in) * 0.5 + frac_vaccinated_in * 0.75 * (1 - effVac)) / (
                1 - frac_vaccinated_in + frac_vaccinated_in * (1 - effVac))

    frac_vaccinated = frac_vaccinated_in * effVac

    # intialize a single exposed case in tile 1
    pars_vac90 = parameters_fit.copy()
    pars_vac90[n_adm * 0:n_adm * 5] = 0
    pars_vac90[0:n_adm] = 1
    pars_vac90[n_adm * 4:n_adm * 5] = np.array(fixed_pars[4]) * frac_vaccinated

    # parameters
    fixed_pars_vac90 = fixed_pars[:4].copy()
    fixed_pars_vac90.append(list(np.array(fixed_pars[4]) * (1 - frac_vaccinated)))
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
def getPar_list(df_params, ind, i_data):
    # Get parameters
    if useTinfUFix:
        TinfUi = 2.
    else:
        TinfUi = df_params['TinfUi'].values[ind]
    Tinc = df_params['Tinc'].values[ind]
    TinfP = df_params['TinfP'].values[ind]
    E0 = df_params['E0'].values[ind]
    if i_data == 0:
        the_title = 'Median Income'
        R_T1 = df_params['R_T1_MedInc'].values[ind]
        R_T2 = df_params['R_T2_MedInc'].values[ind]
        R_T3 = df_params['R_T3_MedInc'].values[ind]
    elif i_data == 1:
        the_title = '1P-Household'
        R_T1 = df_params['R_T1_1P'].values[ind]
        R_T2 = df_params['R_T2_1P'].values[ind]
        R_T3 = df_params['R_T3_1P'].values[ind]
    elif i_data == 2:
        the_title = 'Living Space'
        R_T1 = df_params['R_T1_Liv'].values[ind]
        R_T2 = df_params['R_T2_Liv'].values[ind]
        R_T3 = df_params['R_T3_Liv'].values[ind]
    elif i_data == 3:
        the_title = 'Senior'
        R_T1 = df_params['R_T1_Sen'].values[ind]
        R_T2 = df_params['R_T2_Sen'].values[ind]
        R_T3 = df_params['R_T3_Sen'].values[ind]

    # Add to list
    this_par_list = [0 for i in range(0, 36)]
    this_par_list[18] = Tinc
    this_par_list[34] = TinfP
    this_par_list[28] = TinfUi
    this_par_list[15] = R_T1
    this_par_list[16] = R_T2
    this_par_list[17] = R_T3
    this_par_list[35] = 0.88

    return np.array(this_par_list)
def get_socDF(useForTiles):

    if useForTiles == 'Median income':
        filename = filename_soc_MedInc
    elif useForTiles == 'Seniority':
        filename = filename_soc_Seniority
    elif useForTiles == 'LivingSpace' or 'Living space\nper person':
        filename = filename_soc_LivSp
    elif useForTiles == 'random':
        filename = filename_soc_random
    soc_df = pd.read_csv(filename)

    return soc_df



################################################################################

if __name__ in "__main__":
     main()

