# -*- coding: latin-1 -*-

import os
import inspect
import itertools
from math import sqrt
import pandas as pd
import numpy as np
import scipy.stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
import matplotlib.image as mpimg
from cnwheat import parameters
from cnwheat import tools
from elongwheat import parameters as elongwheat_parameters
import fspmwheat

## ---------------------------------------------------------------------------------------------------------------------------------------
## ----------- IMPORT DATA
## ---------------------------------------------------------------------------------------------------------------------------------------

# TT entre semis et initialisation du modele
TT_since_sowing = 390  # °C.d

# Define plant density (culm m-2)
PLANT_DENSITY = {1: 250}

# directory with results of Ljutovac Simulation
scenario_name = '2_PARi_MA4'
scenario_dirpath = os.path.join(scenario_name)
scenario_graphs_dirpath = os.path.join(scenario_dirpath, 'graphs')
scenario_outputs_dirpath = os.path.join(scenario_dirpath, 'outputs')
scenario_postprocessing_dirpath = os.path.join(scenario_dirpath, 'postprocessing')

# the path of the CSV files where to save the states of the modeled system at each step
OUTPUTS_DIRPATH = scenario_outputs_dirpath
AXES_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'axes_states.csv')
SAM_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'SAM_states.csv')
ORGANS_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'organs_states.csv')
HIDDENZONES_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'hiddenzones_states.csv')
ELEMENTS_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'elements_states.csv')
SOILS_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'soils_states.csv')

# post-processing directory path
POSTPROCESSING_DIRPATH = scenario_postprocessing_dirpath
AXES_POSTPROCESSING_FILEPATH = os.path.join(POSTPROCESSING_DIRPATH, 'axes_postprocessing.csv')
ORGANS_POSTPROCESSING_FILEPATH = os.path.join(POSTPROCESSING_DIRPATH, 'organs_postprocessing.csv')
HIDDENZONES_POSTPROCESSING_FILEPATH = os.path.join(POSTPROCESSING_DIRPATH, 'hiddenzones_postprocessing.csv')
ELEMENTS_POSTPROCESSING_FILEPATH = os.path.join(POSTPROCESSING_DIRPATH, 'elements_postprocessing.csv')
SOILS_POSTPROCESSING_FILEPATH = os.path.join(POSTPROCESSING_DIRPATH, 'soils_postprocessing.csv')

df_axe = pd.read_csv(AXES_POSTPROCESSING_FILEPATH)
df_axe = df_axe[(df_axe.plant == 1) & (df_axe.axis == 'MS')].copy()
df_axe['day'] = df_axe['t'] // 24 + 1

df_hz = pd.read_csv(HIDDENZONES_POSTPROCESSING_FILEPATH)
df_hz = df_hz[(df_hz.plant == 1) & (df_hz.axis == 'MS')].copy()
df_hz['day'] = df_hz['t'] // 24 + 1

df_elt = pd.read_csv(ELEMENTS_POSTPROCESSING_FILEPATH)
df_elt_all = df_elt.copy()
df_elt = df_elt[(df_elt.plant == 1) & (df_elt.axis == 'MS')].copy()
df_elt['day'] = df_elt['t'] // 24 + 1

df_org = pd.read_csv(ORGANS_POSTPROCESSING_FILEPATH)
df_roots = df_org[df_org['organ'] == 'roots'].copy()
df_roots['day'] = df_roots['t'] // 24 + 1
df_phloem = df_org[df_org['organ'] == 'phloem'].copy()
df_phloem['day'] = df_phloem['t'] // 24 + 1

out_sam = pd.read_csv(SAM_STATES_FILEPATH)
out_sam = out_sam[(out_sam.plant == 1) & (out_sam.axis == 'MS')].copy()

out_sam['day'] = out_sam['t'] // 24
out_sam_days = out_sam.groupby(['day']).agg({'sum_TT': 'max'})

out_hz = pd.read_csv(HIDDENZONES_STATES_FILEPATH)
out_hz = out_hz[(out_hz.plant == 1) & (out_hz.axis == 'MS')].copy()

out_elt = pd.read_csv(ELEMENTS_STATES_FILEPATH)
out_elt = out_elt[(out_elt.plant == 1) & (out_elt.axis == 'MS')].copy()

out_soil = pd.read_csv(SOILS_STATES_FILEPATH)

# Meteo
fspm = os.path.join(inspect.getfile(fspmwheat), '..', '..')
METEO_INPUTS_FILEPATH = os.path.join(fspm, 'example', 'Vegetative_stages', 'inputs', 'meteo_Ljutovac2002.csv')
meteo = pd.read_csv(METEO_INPUTS_FILEPATH)
meteo['day'] = meteo['t'] // 24
meteo_days = meteo.groupby(['day']).agg({'PARi': 'sum',
                                         'PARi_MA4': 'sum',
                                         'air_temperature': 'mean',
                                         'soil_temperature': 'mean',
                                         'humidity': 'mean'})

# Observed value from Ljutovac 2002
data_obs = pd.read_csv(os.path.join('Data_Ljutovac', 'Ljutovac2002.csv'))

# Unloadings
scenario_unloadings_dirpath = os.path.join('all_integration_unloadings.csv')
df_load = pd.read_csv(scenario_unloadings_dirpath, sep=',')
df_load5 = df_load[df_load.metamer == 5].copy()
df_load8 = df_load[df_load.metamer == 8].copy()

## ---------------------------------------------------------------------------------------------------------------------------------------
## -----------  USEFULL
## ---------------------------------------------------------------------------------------------------------------------------------------

## Graph parameters

x_name = 't'
x_label = 'Hours since initialization'
x_label_TT = u'Time since leaf 4 emergence (°Cd)'
x_label_days = 'Days since leaf 4 emergence'

colors = ['blue', 'gold', 'green', 'red', 'darkviolet', 'darkorange', 'magenta', 'brown', 'darkcyan', 'grey', 'lime']
colors = colors + colors

plt.rc('font', family='sans-serif')
plt.rc('xtick', labelsize=8)  # 'x-small')
plt.rc('ytick', labelsize=8)  # 'x-small')
plt.rc('axes', labelsize=8, titlesize=10)  #
plt.rc('legend', fontsize=8, frameon=False)  #
plt.rc('lines', markersize=6)


## Functions

def confint(sd, nb):
    1.96 * sd / sqrt(nb)


def expand_grid(data_dict):
    """Create a dataframe from every combination of given values."""
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


# Function : add inset chart into a chart
def add_subplot_axes(ax, rect, axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x, y, width, height], axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2] ** 0.5
    y_labelsize *= rect[3] ** 0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


## ---------------------------------------------------------------------------------------------------------------------------------------
## -----------  GRAPHS
## ---------------------------------------------------------------------------------------------------------------------------------------

## ----------- Meteo and N soil

## -- 0) Meteo DAILY + N soil
out_sam_days_prec = out_sam_days.copy()
out_sam_days_prec['day_next'] = out_sam_days_prec.index + 1
out_sam_days_prec['sum_TT_prec'] = out_sam_days_prec.sum_TT
out_sam_days = out_sam_days.merge(out_sam_days_prec[['day_next', 'sum_TT_prec']], left_on='day', right_on='day_next').copy()
out_sam_days['day'] = out_sam_days.index
out_sam_days['TT'] = out_sam_days.sum_TT - out_sam_days.sum_TT_prec
meteo_days = meteo_days.merge(out_sam_days[['day', 'sum_TT_prec', 'sum_TT', 'TT']], left_on='day', right_on='day').copy()

## PTQ2 using s at 12°c
meteo_days['PTQ2'] = meteo_days.PARi * 3600 * 10 ** -6 / meteo_days.TT / 12
meteo_days.loc[meteo_days.air_temperature < 1, 'PTQ2'] = np.nan
meteo_days.loc[103, 'PTQ2'] = np.nan

## PTQ2 using s at 12°c smooth
meteo_days['TT'] = meteo_days.sum_TT - meteo_days.sum_TT_prec
meteo_days['TT7'] = meteo_days['TT'].rolling(7, min_periods=1).sum()
meteo_days['Tair7'] = meteo_days['air_temperature'].rolling(7, min_periods=1).mean()
meteo_days['PTQ2_smooth'] = (meteo_days['PARi'].rolling(7, min_periods=1).sum() * 3600 * 10 ** -6) / meteo_days['TT7'] / 12

for i in range(0, 5):
    meteo_days.at[i, 'PTQ2_smooth'] = np.nan
meteo_days.at[meteo_days.Tair7 < 2.5, 'PTQ2_smooth'] = np.nan

out_soil['day'] = out_soil['t'] // 24
soil_days = out_soil.groupby(['day']).agg({'nitrates': 'mean'})
soil_days['Unite_N'] = soil_days.nitrates * 14 * 10 ** -6  # kgN/ha

fig = plt.figure(figsize=(4, 12))
gs = gridspec.GridSpec(5, 1, height_ratios=[1, 0.7, 0.7, 0.7, 0.7])

## -- Temperatures
ax0 = plt.subplot(gs[0])
ax0.set_xlim(0, 2500. / 24)
ax0.plot(meteo_days.index, meteo_days.air_temperature, label=r'Air')
ax0.plot(meteo_days.index, meteo_days.soil_temperature, label=r'Soil', color='r')
ax0.legend(loc='upper center', frameon=True)

ax0.get_yaxis().set_label_coords(-0.08, 0.5)
ax0.set_ylabel(u'Temperature (°C)')
plt.setp(ax0.get_xticklabels(), visible=False)
ax0.get_yaxis().set_label_coords(-0.08, 0.5)

## -- SumTT
# ax00 =  plt.subplot(gs[1])
ax00 = ax0.twinx()
ax00.set_xlim(0, 2500. / 24)
ax00.set_ylim(0, 700)
ax00.plot(out_sam_days.index, out_sam_days.sum_TT, label=r'sum_TT', linestyle='--', color='k')

ax00.get_yaxis().set_label_coords(1.12, 0.5)
ax00.set_ylabel(u'Time since leaf 4 emergence (°Cd)')
# plt.setp(ax00.get_xticklabels(), visible=False)


## -- PAR
ax1 = plt.subplot(gs[1], sharex=ax0)
ax1.set_xlim(0, 2500. / 24)
ax1.set_ylim(0, 40.)
ax1.plot(meteo_days.index, meteo_days.PARi_MA4 * 3600 * 10 ** -6, label=r'PARi')
ax1.set_ylabel(u'Incident PAR (mol m$^{-2}$ d$^{-1}$)')
ax1.get_yaxis().set_label_coords(-0.08, 0.5)
plt.setp(ax1.get_xticklabels(), visible=False)
yticks = ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

## -- PTQ
ax2 = plt.subplot(gs[2], sharex=ax0)
ax2.set_xlim(0, 2500. / 24)
ax2.set_ylim(0, 0.4)
ax2.plot(meteo_days.index, meteo_days.PTQ2_smooth)
ax2.set_ylabel(u'Photothermal quotient\n(mol m$^{-2}$ d$^{-1}$ at 12°C)')
ax2.get_yaxis().set_label_coords(-0.08, 0.5)
yticks = ax2.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
plt.setp(ax2.get_xticklabels(), visible=False)
ax2.set_yticks([0, 0.1, 0.2, 0.3])

## -- Humidity
ax3 = plt.subplot(gs[3], sharex=ax0)
ax3.set_ylim(0, 1)
ax3.set_xlim(0, 2500. / 24)
ax3.plot(meteo_days.index, meteo_days.humidity, label=r'Relative humidity')
ax3.set_ylabel(u'Relative humidity')
ax3.get_yaxis().set_label_coords(-0.08, 0.5)
yticks = ax3.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
plt.setp(ax3.get_xticklabels(), visible=False)

## -- soil N
ax4 = plt.subplot(gs[4], sharex=ax0)
ax4.set_ylim(bottom=0, top=10)
ax4.set_xlim(0, 2500. / 24)
ax4.plot(soil_days.index, soil_days.Unite_N, color='r')
ax4.set_ylabel(u'Soil Nitrates (kg N ha$^{-1}$)')
ax4.get_yaxis().set_label_coords(-0.08, 0.5)
yticks = ax4.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
ax4.set_xlabel(x_label_days)

# remove vertical gap between subplots
plt.subplots_adjust(hspace=.0)

# letters
ax0.text(0.08, 0.9, 'A', ha='center', va='center', size=9, transform=ax0.transAxes)
ax1.text(0.08, 0.9, 'B', ha='center', va='center', size=9, transform=ax1.transAxes)
ax2.text(0.08, 0.9, 'C', ha='center', va='center', size=9, transform=ax2.transAxes)
ax3.text(0.08, 0.75, 'D', ha='center', va='center', size=9, transform=ax3.transAxes)
ax4.text(0.08, 0.9, 'E', ha='center', va='center', size=9, transform=ax4.transAxes)

# second x-axis
ax30 = ax4.twiny()
new_tick_locations = np.array([0, 100, 200, 300, 400, 500, 600])
old_tick_locations = np.array([0, 17.8, 38.5, 63.5, 81, 93, 105])

ax30.set_xlim(ax4.get_xlim())
ax30.set_xticks(old_tick_locations)
ax30.set_xticklabels(new_tick_locations)
ax30.set_xlabel(u'Thermal time since leaf 4 emergence (°Cd)')
ax30.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
ax30.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
ax30.spines['bottom'].set_position(('outward', 36))

plt.savefig(os.path.join(scenario_graphs_dirpath, 'Meteo_Nsoil_days.PNG'), format='PNG', bbox_inches='tight', dpi=600)
plt.close()

## -----------  Leaf traits

## -- Final dimensions vs. Ljutovac 2002
res = out_hz[~np.isnan(out_hz.leaf_Lmax)].copy()
res_IN = res[~ np.isnan(res.internode_Lmax)]
last_value_idx = res.groupby(['metamer'])['t'].transform(max) == res['t']
res = res[last_value_idx].copy()
res['lamina_Wmax'] = res.leaf_Wmax
res['lamina_W_Lg'] = res.leaf_Wmax / res.lamina_Lmax
last_value_idx = res_IN.groupby(['metamer'])['t'].transform(max) == res_IN['t']
res_IN = res_IN[last_value_idx].copy()
res = res[res.metamer <= 8].copy()

# Ljutovac observed data
bchmk = data_obs
bchmk = bchmk[(bchmk.metamer <= 8) & (bchmk.metamer >= 3)].copy()
bchmk2 = pd.read_csv(os.path.join('Data_Ljutovac', 'Ljutovac2002_ind.csv'))
bchmk2 = bchmk2[(bchmk2.Phytomer <= 8) & (bchmk2.Phytomer >= 3)].copy()

bchmk2['L_confint'] = 1.96 * bchmk2.L_SD / np.sqrt(bchmk2.L_nb)
bchmk2['Lg_confint'] = 1.96 * bchmk2.Lg_SD / np.sqrt(bchmk2.Lg_nb)

## -- Leaf RER during the exponentiel-like phase
## RER parameters
rer_param = dict((k, v) for k, v in elongwheat_parameters.RERmax.iteritems() if k < 9)

## Simulated RER

# import simulation outputs
data_RER = out_hz[(out_hz.metamer >= 4)].copy()
data_RER.sort_values(['t', 'metamer'], inplace=True)
data_teq = out_sam.copy()

## Time previous leaf emergence
tmp = data_RER[data_RER.leaf_is_emerged]
leaf_em = tmp.groupby('metamer', as_index=False)['t'].min()
leaf_em['t_em'] = leaf_em.t
prev_leaf_em = leaf_em
prev_leaf_em.metamer = leaf_em.metamer + 1

data_RER2 = pd.merge(data_RER, prev_leaf_em[['metamer', 't_em']], on='metamer')
data_RER2 = data_RER2[data_RER2.t <= data_RER2.t_em]

## SumTimeEq
data_teq['SumTimeEq'] = np.cumsum(data_teq.delta_teq)
data_RER3 = pd.merge(data_RER2, data_teq[['t', 'SumTimeEq']], on='t')

## logL
data_RER3['logL'] = np.log(data_RER3.leaf_L)

## Estimate RER
RER_sim = {}
for l in data_RER3.metamer.drop_duplicates():
    if l < 9:
        Y = data_RER3.logL[data_RER3.metamer == l]
        X = data_RER3.SumTimeEq[data_RER3.metamer == l]
        X = sm.add_constant(X)
        mod = sm.OLS(Y, X)
        fit_RER = mod.fit()
        RER_sim[l] = fit_RER.params['SumTimeEq']

## -- Phyllochron
grouped_df = df_hz[df_hz['axis'] == 'MS'].groupby(['plant', 'metamer'])[['t', 'leaf_is_emerged']]
leaf_emergence = {}
for group_name, data in grouped_df:
    plant, metamer = group_name[0], group_name[1]
    if metamer == 3 or True not in data['leaf_is_emerged'].unique(): continue
    leaf_emergence_t = data[data['leaf_is_emerged'] == True].iloc[0]['t']
    leaf_emergence[(plant, metamer)] = leaf_emergence_t

phyllochron = {'plant': [], 'metamer': [], 'phyllochron': []}
leaf_emergence_dd = {'plant': [], 'metamer': [], 'leaf_emergence': []}
for key, leaf_emergence_t in sorted(leaf_emergence.items()):
    plant, metamer = key[0], key[1]
    if metamer == 4: continue
    phyllochron['plant'].append(plant)
    phyllochron['metamer'].append(metamer)
    leaf_emergence_dd['plant'].append(plant)
    leaf_emergence_dd['metamer'].append(metamer)
    prev_leaf_emergence_t = leaf_emergence[(plant, metamer - 1)]
    leaf_emergence_dd_i = out_sam[(out_sam['t'] == leaf_emergence_t)].sum_TT.values[0]
    if out_sam[(out_sam['t'] == leaf_emergence_t) | (out_sam['t'] == prev_leaf_emergence_t)].sum_TT.count() == 2:
        phyllo_DD = leaf_emergence_dd_i - out_sam[(out_sam['t'] == prev_leaf_emergence_t)].sum_TT.values[0]
    else:
        phyllo_DD = np.nan
    phyllochron['phyllochron'].append(phyllo_DD)
    leaf_emergence_dd['leaf_emergence'].append(leaf_emergence_dd_i)

## -- Graph
pos_x_1row = -0.14
pos_x_2row = -0.125
## Longueurs
fig = plt.figure(figsize=(4, 11))
# set height ratios for sublots
gs = gridspec.GridSpec(5, 1, height_ratios=[1.5, 1, 1, 1, 1])

# the fisrt subplot
ax0 = plt.subplot(gs[0])
ax0.set_xlim(2, 10)
ax0.plot(res.metamer, res.leaf_Lmax * 100, label=r'Leaves', linestyle='-', color='g')
ax0.errorbar(bchmk2.Phytomer, bchmk2.L_moy, yerr=bchmk2.L_confint, marker='o', color='g', linestyle='')
ax0.plot(res.metamer, res.sheath_Lmax * 100, label=r'Sheaths', linestyle='-', color='b')
ax0.errorbar(bchmk2.Phytomer, bchmk2.Lg_moy, yerr=bchmk2.Lg_confint, marker='o', color='b', linestyle='')
# align y axis label
ax0.get_yaxis().set_label_coords(pos_x_1row, 0.5)
ax0.set_ylabel(u'Mature length (cm)')

## RER
ax1 = plt.subplot(gs[1], sharex=ax0)
ax1.set_xlim(2, 10)
ax1.set_ylim(bottom=0., top=6e-6)
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))

x, y = zip(*sorted(RER_sim.items()))
ax1.plot(x, y, label=r'Simulated RER', linestyle='-', color='g')
ax1.errorbar(data_obs.loc[data_obs.metamer < 9, 'metamer'], data_obs.loc[data_obs.metamer < 9, 'RER'], yerr=data_obs.loc[data_obs.metamer < 9, 'RER_confint'],
             marker='o', color='g', linestyle='', label="Observed RER")
ax1.plot(rer_param.keys(), rer_param.values(), marker='*', color='k', linestyle='', label="Model parameters")

ax1.get_yaxis().set_label_coords(pos_x_2row, 0.5)
ax1.set_ylabel(u'Relative Elongation Rate\nat 12°C (s$^{-1}$)')
yticks = ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
yticks[0].label1.set_visible(False)

# the second subplot
ax2 = plt.subplot(gs[2], sharex=ax0)
ax2.set_xlim(2, 10)
ax2.set_ylim(0., 2.)
ax2.plot(res.metamer, res.lamina_Wmax * 100, label=r'Lamina', linestyle='-', color='r')
ax2.plot(bchmk.metamer, bchmk.lamina_Wmax, marker='o', color='r', linestyle='')

# align y axis label
ax2.get_yaxis().set_label_coords(pos_x_1row, 0.5)
ax2.set_ylabel(u'Lamina width (cm)')
# remove last tick label for the second subplot
yticks = ax2.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

## rapport Width/length lamina
bchmk['lamina_W_Lg'] = bchmk.lamina_Wmax / bchmk.lamina_Lmax
ax20 = ax2.twinx()
ax20.set_xlim(2, 10)
ax20.set_ylim(0.0, 0.1)
ax20.set_yticks([0.02, 0.05, 0.08])
ax20.plot(bchmk.metamer, bchmk.lamina_W_Lg, marker='o', linestyle='', mec='r', mfc='none')
ax20.plot(res.metamer, res.lamina_W_Lg, linestyle='--', color='r')
# ax20.get_yaxis().set_label_coords(1.12,0.5)
ax20.set_ylabel(u'Width:Length\nlamina ratio')

## Masses surfaciques limbes
ax3 = plt.subplot(gs[3], sharex=ax0)
ax3.set_xlim(2, 10)
ax3.set_ylim(10., 30.)
ax3.plot(res.metamer, res.SSLW, label=r'Lamina', linestyle='-', color='g')

# align y axis label
ax3.get_yaxis().set_label_coords(pos_x_2row, 0.5)
ax3.set_ylabel(u'Specific Structural Lamina\nMass (g m$^{-2}$)')
# remove last tick label for the second subplot
yticks = ax3.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

## Masse linéaire de gaine
ax30 = ax3.twinx()
ax30.set_xlim(2, 10)
ax30.set_ylim(0.0, 1)
ax30.set_yticks([0.25, 0.5, 0.75])
ax30.plot(res.metamer, res.LSSW, linestyle='-', color='b')
# ax30.get_yaxis().set_label_coords(1.12,0.5)
ax30.set_ylabel(u'Structural Linear Sheath\nMass (g m$^{-1}$)')

## Phyllochron
ax4 = plt.subplot(gs[4], sharex=ax0)
ax4.set_xlim(2, 10)
ax4.set_ylim(ymin=0, ymax=160)
ax4.set_yticks([0, 50, 100])
ax4.plot(phyllochron['metamer'], phyllochron['phyllochron'], color='g')
ax4.axhline(y=100, linestyle='-.', color='k')
ax4.get_yaxis().set_label_coords(pos_x_1row, 0.5)
ax4.set_ylabel(u'Phyllochron (°Cd)')
# yticks = ax4.yaxis.get_major_ticks()
# yticks[-1].label1.set_visible(False)

## Formatting
# shared axis X
plt.setp(ax0.get_xticklabels(), visible=False)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)
# remove vertical gap between subplots
plt.subplots_adjust(hspace=.0)
ax4.set_xlabel('Phytomer rank')

ax0.text(0.08, 0.9, 'A', ha='center', va='center', size=9, transform=ax0.transAxes)
ax1.text(0.08, 0.9, 'B', ha='center', va='center', size=9, transform=ax1.transAxes)
ax2.text(0.08, 0.9, 'C', ha='center', va='center', size=9, transform=ax2.transAxes)
ax3.text(0.08, 0.9, 'D', ha='center', va='center', size=9, transform=ax3.transAxes)
ax4.text(0.08, 0.9, 'E', ha='center', va='center', size=9, transform=ax4.transAxes)

plt.savefig(os.path.join(scenario_graphs_dirpath, 'Leaf_traits.PNG'), format='PNG', bbox_inches='tight', dpi=600)
plt.close()

## -----------  Leaf Metabolism Carbon

## -- Times
t_end_5_pif = 57
t_end_8_pif = 97

df_hz_5 = df_hz[df_hz.metamer == 5]
df_hz_8 = df_hz[df_hz.metamer == 8]

# Time of leaf emergence
t_em_5 = min(df_hz_5[df_hz_5.leaf_is_emerged].t)
t_em_8 = min(df_hz_8[df_hz_8.leaf_is_emerged].t)
t_em_4 = min(df_hz[(df_hz.metamer == 4) & (df_hz.leaf_is_emerged)].t)
t_em_7 = min(df_hz[(df_hz.metamer == 7) & (df_hz.leaf_is_emerged)].t)
t_em_6 = min(df_hz[(df_hz.metamer == 6) & (df_hz.leaf_is_emerged)].t)
t_em_9 = min(df_hz[(df_hz.metamer == 9) & (df_hz.leaf_is_emerged)].t)

# Time end of leaf elongation
t_end_5 = min(out_elt[(out_elt.metamer == 5) & (out_elt.organ == 'sheath') & (out_elt.is_growing == 0)].t)
t_end_8 = min(out_elt[(out_elt.metamer == 8) & (out_elt.organ == 'blade') & (out_elt.is_growing == 0)].t)  # TODO: run longer simulation

## -- Loadings
df_load5['day'] = df_load5['t_sim'] // 24 + 1
df_load5_days = df_load5.groupby(['day']).agg({'unloading_sucrose_int': 'sum',
                                               'unloading_aa_int': 'sum'})
df_load5_days['day'] = df_load5_days.index

df_load8['day'] = df_load8['t_sim'] // 24 + 1
df_load8_days = df_load8.groupby(['day']).agg({'unloading_sucrose_int': 'sum',
                                               'unloading_aa_int': 'sum'})
df_load8_days['day'] = df_load8_days.index

ga5 = df_elt[(df_elt.element == 'LeafElement1') & (df_elt.metamer == 5)]
ga5_days = ga5.groupby(['day'])['green_area'].mean()
ga8 = df_elt[(df_elt.element == 'LeafElement1') & (df_elt.metamer == 8)]
ga8_days = ga8.groupby(['day'])['green_area'].mean()

## -- Growth costs
df_elt['sum_respi_tillers'] = df_elt['sum_respi'] * df_elt['nb_replications']
df_elt['Net_Photosynthesis_tillers'] = df_elt['Photosynthesis'] * df_elt['nb_replications'] - df_elt['sum_respi_tillers']
df_hz['Respi_growth_tillers'] = df_hz['Respi_growth'] * df_hz['nb_replications']
df_hz['Growth_Costs_C_tillers'] = df_hz['sucrose_consumption_mstruct'] * df_hz['nb_replications'] + df_hz['Respi_growth_tillers']
df_hz['Growth_Costs_N_tillers'] = df_hz['AA_consumption_mstruct'] * df_hz['nb_replications']

df_elt['Amino_Acids_import_tillers'] = df_elt['Amino_Acids_import'] * df_elt['nb_replications']
df_elt['Nitrates_import_tillers'] = df_elt['Nitrates_import'] * df_elt['nb_replications']
df_elt_ms = df_elt.loc[df_elt['axis'] == 'MS']  # df_elt.loc[ df_elt['organ'].isin(['sheath','blade']) & (df_elt['axis# '] == 'MS') ]
Net_Photosynthesis_leaves = df_elt_ms.groupby(['plant', 'metamer', 't'], as_index=False).agg({'Net_Photosynthesis_tillers': 'sum',
                                                                                              'Amino_Acids_import_tillers': 'sum',
                                                                                              'Nitrates_import_tillers': 'sum'})
Net_Photosynthesis_leaves['Net_Photosynthesis_leaves_cum'] = Net_Photosynthesis_leaves.groupby(['plant', 'metamer'])['Net_Photosynthesis_tillers'].cumsum()
Net_Photosynthesis_leaves['AA_import_leaves_cum'] = Net_Photosynthesis_leaves.groupby(['plant', 'metamer'])['Amino_Acids_import_tillers'].cumsum()
Net_Photosynthesis_leaves['Nitrates_import_cum'] = Net_Photosynthesis_leaves.groupby(['plant', 'metamer'])['Nitrates_import_tillers'].cumsum()
Net_Photosynthesis_leaves['N_import_cum'] = Net_Photosynthesis_leaves['Nitrates_import_cum'] + Net_Photosynthesis_leaves['AA_import_leaves_cum']

Net_Photosynthesis_leaves['day'] = Net_Photosynthesis_leaves['t'] // 24 + 1
Net_Photosynthesis_5_days = Net_Photosynthesis_leaves[Net_Photosynthesis_leaves.metamer == 5].groupby(['day']).agg({'Net_Photosynthesis_leaves_cum': 'max'})
Net_Photosynthesis_8_days = Net_Photosynthesis_leaves[Net_Photosynthesis_leaves.metamer == 8].groupby(['day']).agg({'Net_Photosynthesis_leaves_cum': 'max'})
N_imports_5_days = Net_Photosynthesis_leaves[Net_Photosynthesis_leaves.metamer == 5].groupby(['day']).agg({'N_import_cum': 'max'})
N_imports_8_days = Net_Photosynthesis_leaves[Net_Photosynthesis_leaves.metamer == 8].groupby(['day']).agg({'N_import_cum': 'max'})
Nitrates_imports_5_days = Net_Photosynthesis_leaves[Net_Photosynthesis_leaves.metamer == 5].groupby(['day']).agg({'Nitrates_import_cum': 'max'})
Nitrates_imports_8_days = Net_Photosynthesis_leaves[Net_Photosynthesis_leaves.metamer == 8].groupby(['day']).agg({'Nitrates_import_cum': 'max'})

Growth_Costs_C = df_hz.groupby(['plant', 'metamer', 't'], as_index=False).agg({'Growth_Costs_C_tillers': 'sum',
                                                                               'Growth_Costs_N_tillers': 'sum'})
Growth_Costs_C['Growth_Costs_C_cum'] = Growth_Costs_C.groupby(['plant', 'metamer'])['Growth_Costs_C_tillers'].cumsum()
Growth_Costs_C['Growth_Costs_N_cum'] = Growth_Costs_C.groupby(['plant', 'metamer'])['Growth_Costs_N_tillers'].cumsum()
Growth_Costs_C['day'] = Growth_Costs_C['t'] // 24 + 1
Growth_Costs_C5_days = Growth_Costs_C[Growth_Costs_C.metamer == 5].groupby(['day']).agg({'Growth_Costs_C_cum': 'max'})
Growth_Costs_C8_days = Growth_Costs_C[Growth_Costs_C.metamer == 8].groupby(['day']).agg({'Growth_Costs_C_cum': 'max'})
Growth_Costs_N5_days = Growth_Costs_C[Growth_Costs_C.metamer == 5].groupby(['day']).agg({'Growth_Costs_N_cum': 'max'})
Growth_Costs_N8_days = Growth_Costs_C[Growth_Costs_C.metamer == 8].groupby(['day']).agg({'Growth_Costs_N_cum': 'max'})

## -- Storage
df_hz_5['day'] = df_hz_5['t'].copy(deep=True) // 24 + 1
df_hz_5_days = df_hz_5.groupby(['day']).agg({'Cont_Proteins_DM': 'mean',
                                             'Cont_Fructan_DM': 'mean',
                                             'internode_L': 'max',
                                             'leaf_L': 'max'})
df_hz_8['day'] = df_hz_8['t'].copy(deep=True) // 24 + 1
df_hz_8_days = df_hz_8.groupby(['day']).agg({'Cont_Proteins_DM': 'mean',
                                             'Cont_Fructan_DM': 'mean',
                                             'internode_L': 'max',
                                             'leaf_L': 'max'})
## -- Graph Metabolism C

fig = plt.figure(figsize=(8, 9))
# set height ratios for sublots
gs = gridspec.GridSpec(3, 2)

ax0 = plt.subplot(gs[0])
ax0.plot(df_load5_days.day[df_load5_days.day < t_end_5_pif], df_load5_days.unloading_sucrose_int[df_load5_days.day < t_end_5_pif] / 12, label=u'Sucrose unloadings (µmol.d$^{-1}$)', color='b')
# ax0.plot(df_load5_days.day[df_load5_days.day < t_end_5_pif], df_load5_days.unloading_aa_int[df_load5_days.day <t_end_5_pif],label = u'Amino acids unloadings (µmol N.d$^{-1}$)' , color = 'r')
ax0.axhline(0, color='k')
ax0.axvline(t_em_5 // 24 + 1, color='k', linestyle='--')
ax0.axvline(t_em_4 // 24 + 1, color='k', linestyle=':')
ax0.set_xlim(0, 120)
ax0.set_ylim(-15, 15)
ax0.get_yaxis().set_label_coords(-0.1, 0.5)
ax0.set_ylabel(u'Sucrose unloading into $\it{hz}$ (µmol d$^{-1}$)')
plt.setp(ax0.get_xticklabels(), visible=False)

## Unloadings F8
ax1 = plt.subplot(gs[1], sharey=ax0)
line_unload_C, = ax1.plot(df_load8_days.day[df_load8_days.day < t_end_8_pif], df_load8_days.unloading_sucrose_int[df_load8_days.day < t_end_8_pif] / 12, label=u'Sucrose unloadings (µmol.d$^{-1}$)',
                          color='b')
# line_unload_N, = ax1.plot(df_load8_days.day[df_load8_days.day < t_end_8_pif], df_load8_days.unloading_aa_int[df_load8_days.day < t_end_8_pif], label = u'Amino acids unloadings (µmol N.d$^{-1}$)' , color = 'r')
ax1.axhline(0, color='k')
ax1.axvline(t_em_8 // 24 + 1, color='k', linestyle='--')
ax1.axvline(t_em_7 // 24 + 1, color='k', linestyle=':')
ax1.set_xlim(0, 120)
ax1.set_ylim(-15, 15)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax1.get_yticklabels(), visible=False)

## Growth_Costs_C5_days and Net_Photosynthesis_5_days
ax2 = plt.subplot(gs[2], sharex=ax0)
line_costC, = ax2.plot(Growth_Costs_C5_days.index, Growth_Costs_C5_days.Growth_Costs_C_cum * 10 ** -3, linestyle='-', label=u'C used for structural growth', color='m')
line_prodC, = ax2.plot(Net_Photosynthesis_5_days.index, Net_Photosynthesis_5_days.Net_Photosynthesis_leaves_cum * 10 ** -3, linestyle='-', label=u'C produced by photosynthesis', color='g')
ax2.set_ylim(bottom=0., top=30)
ax2.get_yaxis().set_label_coords(-0.1, 0.5)
ax2.set_ylabel(u'Carbon (mmol)')
plt.setp(ax2.get_xticklabels(), visible=False)
ax2.axvline(t_em_5 // 24 + 1, color='k', linestyle='--')
ax2.axvline(t_em_4 // 24 + 1, color='k', linestyle=':')

## Growth_Costs_C8_days and Net_Photosynthesis_8_days
ax3 = plt.subplot(gs[3], sharex=ax1, sharey=ax2)
ax3.plot(Growth_Costs_C8_days.index, Growth_Costs_C8_days.Growth_Costs_C_cum * 10 ** -3, linestyle='-', label=u'C used for structural growth', color='m')
ax3.plot(Net_Photosynthesis_8_days.index, Net_Photosynthesis_8_days.Net_Photosynthesis_leaves_cum * 10 ** -3, linestyle='-', label=u'C produced by photosynthesis', color='g')
ax3.set_ylim(bottom=0., top=30)
ax3.axvline(t_em_8 // 24 + 1, color='k', linestyle='--')
ax3.axvline(t_em_7 // 24 + 1, color='k', linestyle=':')
plt.setp(ax3.get_xticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)
yticks = ax2.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

## Length F5
ax4 = plt.subplot(gs[4], sharex=ax2)
ax4.plot(df_hz_5_days.index, df_hz_5_days.leaf_L * 100, label=r'Leaf', linestyle='-', color='k')
# ax4.plot(df_hz_5_days.index, df_hz_5_days.internode_L*100, label=r'Internode', linestyle='--', color = 'k')
ax4.set_ylim(bottom=0., top=40)
yticks = ax4.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
ax4.get_yaxis().set_label_coords(-0.1, 0.5)
ax4.set_ylabel(u'Leaf length (cm)')
# plt.setp(ax4.get_xticklabels(), visible=False)
ax4.axvline(t_em_5 // 24 + 1, color='k', linestyle='--')
ax4.axvline(t_em_4 // 24 + 1, color='k', linestyle=':')

ax40 = ax4.twinx()
ax40.plot(ga5_days.index, ga5_days * 10000, label=r'Lamina 5', linestyle='--', color='k')
ax40.set_ylim(0, 40)
plt.setp(ax40.get_yticklabels(), visible=False)

## Length F8
ax5 = plt.subplot(gs[5], sharex=ax3)
ax5.plot(df_hz_8_days.index, df_hz_8_days.leaf_L * 100, label=r'Leaf', linestyle='-', color='k')
# ax5.plot(df_hz_8_days.index, df_hz_8_days.internode_L*100, label=r'Internode', linestyle='--', color = 'k')
ax5.set_ylim(bottom=0., top=40)
# plt.setp(ax5.get_xticklabels(), visible=False)
plt.setp(ax5.get_yticklabels(), visible=False)
ax5.axvline(t_em_8 // 24 + 1, color='k', linestyle='--')
ax5.axvline(t_em_7 // 24 + 1, color='k', linestyle=':')

ax50 = ax5.twinx()
ax50.plot(ga8_days.index, ga8_days * 10000, label=r'Lamina 8', linestyle='--', color='k')
ax50.get_yaxis().set_label_coords(1.1, 0.5)
ax50.set_ylabel(u'Exposed lamina area (cm$^{-2}$)')
ax50.set_ylim(0, 40)
yticks = ax50.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

# axis X
ax4.set_xlabel(x_label_days)
ax5.set_xlabel(x_label_days)

# remove vertical gap between subplots
plt.subplots_adjust(hspace=.0, wspace=.0)

# Titles
plt.setp(ax0, title='Phytomer 5')
plt.setp(ax1, title='Phytomer 8')

# put legends
ax2.legend((line_costC, line_prodC),
           (u'Consumption for\nstructural growth', 'Production'),
           loc='upper right',  # bbox_to_anchor=(1, 0.5),
           frameon=True)
# letters by subplots
ax0.text(0.08, 0.9, 'A', ha='center', va='center', size=9, transform=ax0.transAxes)
ax1.text(0.08, 0.9, 'B', ha='center', va='center', size=9, transform=ax1.transAxes)
ax2.text(0.08, 0.9, 'C', ha='center', va='center', size=9, transform=ax2.transAxes)
ax3.text(0.08, 0.9, 'D', ha='center', va='center', size=9, transform=ax3.transAxes)
ax4.text(0.08, 0.9, 'E', ha='center', va='center', size=9, transform=ax4.transAxes)
ax5.text(0.08, 0.9, 'F', ha='center', va='center', size=9, transform=ax5.transAxes)

plt.savefig(os.path.join(scenario_graphs_dirpath, 'Leaf_Metabolism_C.PNG'), format='PNG', bbox_inches='tight', dpi=600)
plt.close()

## -----------   Leaf Metabolism N

fig = plt.figure(figsize=(8, 9))
# set height ratios for sublots
gs = gridspec.GridSpec(3, 2)

ax0 = plt.subplot(gs[0])
# ax0.plot(df_load5_days.day[df_load5_days.day < t_end_5_pif ], df_load5_days.unloading_sucrose_int[df_load5_days.day < t_end_5_pif]/12,label = u'Sucrose unloadings (µmol.d$^{-1}$)', color='b' )
ax0.plot(df_load5_days.day[df_load5_days.day < t_end_5_pif], df_load5_days.unloading_aa_int[df_load5_days.day < t_end_5_pif], label=u'Amino acids unloadings (µmol N.d$^{-1}$)', color='r')
ax0.axhline(0, color='k')
ax0.axvline(t_em_5 // 24 + 1, color='k', linestyle='--')
ax0.axvline(t_em_4 // 24 + 1, color='k', linestyle=':')
ax0.set_xlim(0, 120)
ax0.set_ylim(-15, 15)
ax0.get_yaxis().set_label_coords(-0.1, 0.5)
ax0.set_ylabel(u'Amino acids unloading into $\it{hz}$ (µmol N d$^{-1}$)')
plt.setp(ax0.get_xticklabels(), visible=False)

## Unloadings F8
ax1 = plt.subplot(gs[1], sharey=ax0)
# line_unload_C, = ax1.plot(df_load8_days.day[df_load8_days.day < t_end_8_pif], df_load8_days.unloading_sucrose_int[df_load8_days.day < t_end_8_pif]/12, label = u'Sucrose unloadings (µmol.d$^{-1}$)' , color = 'b')
line_unload_N, = ax1.plot(df_load8_days.day[df_load8_days.day < t_end_8_pif], df_load8_days.unloading_aa_int[df_load8_days.day < t_end_8_pif], label=u'Amino acids unloadings (µmol N.d$^{-1}$)',
                          color='r')
ax1.axhline(0, color='k')
ax1.axvline(t_em_8 // 24 + 1, color='k', linestyle='--')
ax1.axvline(t_em_7 // 24 + 1, color='k', linestyle=':')
ax1.set_xlim(0, 120)
ax1.set_ylim(-15, 15)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax1.get_yticklabels(), visible=False)

## N phytomer 5
ax2 = plt.subplot(gs[2], sharex=ax0)
ax2.plot(Nitrates_imports_5_days.index, Nitrates_imports_5_days.Nitrates_import_cum, color='g')
ax2.plot(Growth_Costs_N5_days.index, Growth_Costs_N5_days.Growth_Costs_N_cum, color='m')
ax2.set_ylim(bottom=0., top=500)
yticks = ax2.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
plt.setp(ax2.get_xticklabels(), visible=False)
ax2.get_yaxis().set_label_coords(-0.1, 0.5)
ax2.set_ylabel(u'Nitrogen (µmol)')

rect = [0.1, 0.3, 0.45, 0.45]
ax20 = add_subplot_axes(ax2, rect)
ax20.plot(Nitrates_imports_5_days.index, Nitrates_imports_5_days.Nitrates_import_cum, color='g')
ax20.plot(Growth_Costs_N5_days.index, Growth_Costs_N5_days.Growth_Costs_N_cum, color='m')
ax20.set_ylim(0, 10)
ax20.set_xlim(0, 50)

## N phytomer 8
ax3 = plt.subplot(gs[3], sharex=ax1, sharey=ax2)
ax3.plot(Nitrates_imports_8_days.index, Nitrates_imports_8_days.Nitrates_import_cum, color='g')
ax3.plot(Growth_Costs_N8_days.index, Growth_Costs_N8_days.Growth_Costs_N_cum, color='m')
ax3.set_ylim(bottom=0., top=500)
plt.setp(ax3.get_yticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)

rect = [0.1, 0.3, 0.45, 0.45]
ax30 = add_subplot_axes(ax3, rect)
ax30.plot(Nitrates_imports_8_days.index, Nitrates_imports_8_days.Nitrates_import_cum, color='g')
ax30.plot(Growth_Costs_N8_days.index, Growth_Costs_N8_days.Growth_Costs_N_cum, color='m')
ax30.set_xlim(50, 100)
ax30.set_ylim(0, 30)

ax2.axvline(t_em_5 // 24 + 1, color='k', linestyle='--')
ax2.axvline(t_em_4 // 24 + 1, color='k', linestyle=':')
ax3.axvline(t_em_8 // 24 + 1, color='k', linestyle='--')
ax3.axvline(t_em_7 // 24 + 1, color='k', linestyle=':')

## Length F5
ax4 = plt.subplot(gs[4], sharex=ax2)
ax4.plot(df_hz_5_days.index, df_hz_5_days.leaf_L * 100, label=r'Leaf', linestyle='-', color='k')
# ax4.plot(df_hz_5_days.index, df_hz_5_days.internode_L*100, label=r'Internode', linestyle='--', color = 'k')
ax4.set_ylim(bottom=0., top=40)
yticks = ax4.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
ax4.get_yaxis().set_label_coords(-0.1, 0.5)
ax4.set_ylabel(u'Leaf length (cm)')
# plt.setp(ax4.get_xticklabels(), visible=False)
ax4.axvline(t_em_5 // 24 + 1, color='k', linestyle='--')
ax4.axvline(t_em_4 // 24 + 1, color='k', linestyle=':')

ax40 = ax4.twinx()
ax40.plot(ga5_days.index, ga5_days * 10000, label=r'Lamina 5', linestyle='--', color='k')
ax40.set_ylim(0, 40)
plt.setp(ax40.get_yticklabels(), visible=False)

## Length F8
ax5 = plt.subplot(gs[5], sharex=ax3)
ax5.plot(df_hz_8_days.index, df_hz_8_days.leaf_L * 100, label=r'Leaf', linestyle='-', color='k')
# ax5.plot(df_hz_8_days.index, df_hz_8_days.internode_L*100, label=r'Internode', linestyle='--', color = 'k')
ax5.set_ylim(bottom=0., top=40)
# plt.setp(ax5.get_xticklabels(), visible=False)
plt.setp(ax5.get_yticklabels(), visible=False)
ax5.axvline(t_em_8 // 24 + 1, color='k', linestyle='--')
ax5.axvline(t_em_7 // 24 + 1, color='k', linestyle=':')

ax50 = ax5.twinx()
ax50.plot(ga8_days.index, ga8_days * 10000, label=r'Lamina 8', linestyle='--', color='k')
ax50.get_yaxis().set_label_coords(1.1, 0.5)
ax50.set_ylabel(u'Exposed lamina area (cm$^{-2}$)')
ax50.set_ylim(0, 40)
yticks = ax50.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

# axis X
ax4.set_xlabel(x_label_days)
ax5.set_xlabel(x_label_days)

# remove vertical gap between subplots
plt.subplots_adjust(hspace=.0, wspace=.0)

# Titles
plt.setp(ax0, title='Phytomer 5')
plt.setp(ax1, title='Phytomer 8')

# put legends
ax2.legend((line_costC, line_prodC),
           (u'Consumption for\nstructural growth', 'Production'),
           loc='upper right',  # bbox_to_anchor=(1, 0.5),
           frameon=True)
# letters by subplots
ax0.text(0.08, 0.9, 'A', ha='center', va='center', size=9, transform=ax0.transAxes)
ax1.text(0.08, 0.9, 'B', ha='center', va='center', size=9, transform=ax1.transAxes)
ax2.text(0.08, 0.9, 'C', ha='center', va='center', size=9, transform=ax2.transAxes)
ax3.text(0.08, 0.9, 'D', ha='center', va='center', size=9, transform=ax3.transAxes)
ax4.text(0.08, 0.9, 'E', ha='center', va='center', size=9, transform=ax4.transAxes)
ax5.text(0.08, 0.9, 'F', ha='center', va='center', size=9, transform=ax5.transAxes)

plt.savefig(os.path.join(scenario_graphs_dirpath, 'Leaf_Metabolism_N.PNG'), format='PNG', bbox_inches='tight', dpi=600)
plt.close()

## -----------   Source/sink relationships

## -- Table C usages relatif to Net Photosynthesis

AMINO_ACIDS_C_RATIO = 4.15  #: Mean number of mol of C in 1 mol of the major amino acids of plants (Glu, Gln, Ser, Asp, Ala, Gly)
AMINO_ACIDS_N_RATIO = 1.25  #: Mean number of mol of N in 1 mol of the major amino acids of plants (Glu, Gln, Ser, Asp, Ala, Gly)

# Photosynthesis
df_elt['Photosynthesis_tillers'] = df_elt['Photosynthesis'].fillna(0) * df_elt['nb_replications'].fillna(1.)
Tillers_Photosynthesis_Ag = df_elt.groupby(['t'], as_index=False).agg({'Photosynthesis_tillers': 'sum'})
C_usages = pd.DataFrame({'t': Tillers_Photosynthesis_Ag['t']})
C_usages['C_produced'] = np.cumsum(Tillers_Photosynthesis_Ag.Photosynthesis_tillers)

# Respiration
C_usages['Respi_roots'] = np.cumsum(df_axe.C_respired_roots)
C_usages['Respi_shoot'] = np.cumsum(df_axe.C_respired_shoot)

# Exudation
C_usages['exudation'] = np.cumsum(df_axe.C_exudated.fillna(0))

# Structural growth
C_consumption_mstruct_roots = df_roots.sucrose_consumption_mstruct.fillna(0) + df_roots.AA_consumption_mstruct.fillna(0) * AMINO_ACIDS_C_RATIO / AMINO_ACIDS_N_RATIO
C_usages['Structure_roots'] = np.cumsum(C_consumption_mstruct_roots.reset_index(drop=True))

df_hz['C_consumption_mstruct'] = df_hz.sucrose_consumption_mstruct.fillna(0) + df_hz.AA_consumption_mstruct.fillna(0) * AMINO_ACIDS_C_RATIO / AMINO_ACIDS_N_RATIO
df_hz['C_consumption_mstruct_tillers'] = df_hz['C_consumption_mstruct'] * df_hz['nb_replications']
C_consumption_mstruct_shoot = df_hz.groupby(['t'])['C_consumption_mstruct_tillers'].sum()
C_usages['Structure_shoot'] = np.cumsum(C_consumption_mstruct_shoot.reset_index(drop=True))

# Non structural C
df_phloem['C_NS'] = df_phloem.sucrose.fillna(0) + df_phloem.amino_acids.fillna(0) * AMINO_ACIDS_C_RATIO / AMINO_ACIDS_N_RATIO
C_NS_phloem_init = df_phloem.C_NS - df_phloem.C_NS[0]
C_usages['NS_phloem'] = C_NS_phloem_init.reset_index(drop=True)

df_elt['C_NS'] = df_elt.sucrose.fillna(0) + df_elt.fructan.fillna(0) + df_elt.starch.fillna(0) + (
        df_elt.amino_acids.fillna(0) + df_elt.proteins.fillna(0)) * AMINO_ACIDS_C_RATIO / AMINO_ACIDS_N_RATIO
df_elt['C_NS_tillers'] = df_elt['C_NS'] * df_elt['nb_replications'].fillna(1.)
C_elt = df_elt.groupby(['t']).agg({'C_NS_tillers': 'sum'})

df_hz['C_NS'] = df_hz.sucrose.fillna(0) + df_hz.fructan.fillna(0) + (df_hz.amino_acids.fillna(0) + df_hz.proteins.fillna(0)) * AMINO_ACIDS_C_RATIO / AMINO_ACIDS_N_RATIO
df_hz['C_NS_tillers'] = df_hz['C_NS'] * df_hz['nb_replications'].fillna(1.)
C_hz = df_hz.groupby(['t']).agg({'C_NS_tillers': 'sum'})

df_roots['C_NS'] = df_roots.sucrose.fillna(0) + df_roots.amino_acids.fillna(0) * AMINO_ACIDS_C_RATIO / AMINO_ACIDS_N_RATIO

C_NS_autre = df_roots.C_NS.reset_index(drop=True) + C_elt.C_NS_tillers + C_hz.C_NS_tillers
C_NS_autre_init = C_NS_autre - C_NS_autre[0]
C_usages['NS_other'] = C_NS_autre_init.reset_index(drop=True)

shoot_plant_mstruct_ratio = df_axe.shoot_roots_mstruct_ratio / (1 + df_axe.shoot_roots_mstruct_ratio)
C_NS_shoot = C_elt.C_NS_tillers + C_hz.C_NS_tillers + df_phloem.C_NS.reset_index(drop=True) * shoot_plant_mstruct_ratio
C_NS_shoot_init = C_NS_shoot - C_NS_shoot[0]
C_usages['NS_shoot'] = C_NS_shoot_init.reset_index(drop=True)
C_NS_roots = df_roots.C_NS.reset_index(drop=True) + df_phloem.C_NS.reset_index(drop=True) * (1 - shoot_plant_mstruct_ratio)
C_NS_roots_init = C_NS_roots - C_NS_roots[0]
C_usages['NS_roots'] = C_NS_roots_init.reset_index(drop=True)

# Total
C_usages['C_budget'] = (C_usages.Respi_roots + C_usages.Respi_shoot + C_usages.exudation + C_usages.Structure_roots + C_usages.Structure_shoot + C_usages.NS_phloem + C_usages.NS_other) / \
                       C_usages.C_produced
# Daily values
C_usages['day'] = C_usages['t'] // 24 + 1
C_usages_days = C_usages.groupby(['day'], as_index=False).mean()

## --- Comparison C fluxes vs. phloem content and vs. photosynthesis
Tillers_Photosynthesis_Ag_day = df_elt.groupby(['day'], as_index=False).agg({'Photosynthesis_tillers': 'sum'})
df_unload = pd.DataFrame({'day': Tillers_Photosynthesis_Ag_day['day']})
df_unload['Tillers_Photosynthesis_Ag'] = Tillers_Photosynthesis_Ag_day.Photosynthesis_tillers
df_unload['sucrose_phloem'] = df_phloem.groupby(['day'], as_index=False).agg({'sucrose': 'mean'}).sucrose

df_load['day'] = df_load['t_sim'] // 24 + 1
df_load['export_sucrose'] = df_load.unloading_sucrose_int.copy(deep=True) * df_load.nb_replications
df_load.loc[df_load.export_sucrose < 0, 'export_sucrose'] = 0
df_load['import_sucrose'] = (df_load.unloading_sucrose_int.copy(deep=True) * df_load.nb_replications) * -1
df_load.loc[df_load.import_sucrose < 0, 'import_sucrose'] = 0
df_load_hz = df_load[df_load.comp == 'hz '].copy(deep=True)
df_load_roots = df_load[df_load.comp == 'roots '].copy(deep=True)
df_load_roots = df_load_roots.merge(df_roots, left_on='t_sim', right_on='t')
df_load_roots['export_sucrose_g'] = df_load_roots.export_sucrose * df_load_roots.mstruct  # l'unité des unloading n'est pas la meme pour les roots que pour les HZ !!

df_unload['export_hz'] = df_load_hz.groupby(['day'], as_index=False).agg({'export_sucrose': 'sum'}).export_sucrose
df_unload['export_roots'] = df_load_roots.groupby(['day_x'], as_index=False).agg({'export_sucrose_g': 'sum'}).export_sucrose_g

df_elt['import_sucrose'] = df_elt.Loading_Sucrose.copy(deep=True) * df_elt.nb_replications
df_elt.loc[df_elt.import_sucrose < 0, 'import_sucrose'] = 0
df_vis_elt = df_elt[(df_elt['element'].isin(['LeafElement1', 'StemElement'])) & (df_elt.is_growing == 0)].copy(deep=True)
df_unload['import_elt'] = df_vis_elt.groupby(['day'], as_index=False).agg({'import_sucrose': 'sum'}).import_sucrose
df_unload['import_hz'] = df_load_hz.groupby(['day'], as_index=False).agg({'import_sucrose': 'sum'}).import_sucrose
df_unload['import_photosynthetic'] = df_unload['import_elt'] + df_unload['import_hz']

## ---  Graph_C_source_sink

fig = plt.figure(figsize=(4, 9))
# set height ratios for sublots
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])

ax = plt.subplot(gs[0])
ax.plot(C_usages_days.day, (C_usages_days.Structure_shoot + C_usages_days.NS_shoot) / C_usages_days.C_produced * 100,
        label=u'Shoot', color='g')
ax.plot(C_usages_days.day, (C_usages_days.Structure_shoot) / C_usages_days.C_produced * 100,
        label=u'Shoot - Structural C only', color='g', linestyle=':')
ax.plot(C_usages_days.day, (C_usages_days.Structure_roots + C_usages_days.NS_roots) / C_usages_days.C_produced * 100,
        label=u'Roots', color='r')
ax.plot(C_usages_days.day, (C_usages_days.Structure_roots) / C_usages_days.C_produced * 100,
        label=u'Roots - Structural C only', color='r', linestyle=':')
ax.plot(C_usages_days.day, (C_usages_days.Respi_roots + C_usages_days.Respi_shoot) / C_usages_days.C_produced * 100, label=u'C loss by respiration', color='b')
ax.plot(C_usages_days.day, C_usages_days.exudation / C_usages_days.C_produced * 100, label=u'C loss by exudation', color='c')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)

ax.get_yaxis().set_label_coords(-0.12, 0.5)
ax.set_ylabel(u'Carbon usages : Photosynthesis (%)')
ax.set_ylim(bottom=0, top=100.)
plt.setp(ax.get_xticklabels(), visible=False)

## loadings phloem vs phloem content
df_unload['puit_hz'] = -1 * df_unload.export_hz / df_unload.sucrose_phloem
df_unload['puit_hz_smooth'] = df_unload.puit_hz.rolling(7).mean()
df_unload['puit_roots'] = -1 * df_unload.export_roots / df_unload.sucrose_phloem
df_unload['puit_roots_smooth'] = df_unload.puit_roots.rolling(7).mean()
df_unload['sce_elt'] = df_unload.import_photosynthetic / df_unload.sucrose_phloem
df_unload['sce_elt_smooth'] = df_unload.sce_elt.rolling(7).mean()

ax1 = plt.subplot(gs[1], sharex=ax)
ax1.plot(df_unload.day, df_unload.puit_hz_smooth * 100,
         label=u'Imports $\it{hz}$', color='g')
ax1.plot(df_unload.day, df_unload.puit_roots_smooth * 100,
         label=u'Imports roots', color='r')
ax1.plot(df_unload.day, df_unload.sce_elt_smooth * 100,
         label=u'Exports photosynthetic elements', color='k')
ax1.get_yaxis().set_label_coords(-0.1, 0.5)
ax1.set_ylabel(u'Sucrose loading into the $\it{phloem}$ :\nSucrose $\it{phloem}$ content (%)')
ax1.set_ylim(bottom=-100, top=100.)
ax1.axhline(y=0, linestyle='-', color='k', linewidth=0.5)
plt.setp(ax1.get_xticklabels(), visible=False)
yticks = ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

ax1.legend(loc='lower left', bbox_to_anchor=(1, -0.15), frameon=True)

## loadings phloem vs net photosynthesis
df_unload['puit_hz2'] = -1 * df_unload.export_hz / df_unload.Tillers_Photosynthesis_Ag
df_unload['puit_hz2_smooth'] = df_unload.puit_hz2.rolling(7).mean()
df_unload['puit_roots2'] = -1 * df_unload.export_roots / df_unload.Tillers_Photosynthesis_Ag
df_unload['puit_roots2_smooth'] = df_unload.puit_roots2.rolling(7).mean()
df_unload['sce_elt2'] = df_unload.import_photosynthetic / df_unload.Tillers_Photosynthesis_Ag
df_unload['sce_elt2_smooth'] = df_unload.sce_elt2.rolling(7).mean()

ax2 = plt.subplot(gs[2], sharex=ax)
ax2.plot(df_unload.day, df_unload.puit_hz2_smooth * 100,
         label=u'Imports $\it{hz}$', color='g')
ax2.plot(df_unload.day, df_unload.puit_roots2_smooth * 100,
         label=u'Imports roots', color='r')
ax2.plot(df_unload.day, df_unload.sce_elt2_smooth * 100,
         label=u'Exports photosynthetic elements', color='k')
ax2.get_yaxis().set_label_coords(-0.1, 0.5)
ax2.set_ylabel(u'Sucrose loading into the $\it{phloem}$ :\nDaily photosynthesis (%)')
ax2.set_ylim(bottom=-100, top=100.)
ax2.axhline(y=0, linestyle='-', color='k', linewidth=0.5)
yticks = ax2.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
ax2.set_xlabel(x_label_days)

## Formatting
# shared axis X
plt.setp(ax0.get_xticklabels(), visible=False)
plt.setp(ax1.get_xticklabels(), visible=False)
# remove vertical gap between subplots
plt.subplots_adjust(hspace=.0)

# arrows
ax.arrow(t_em_5 // 24 + 1, 90, 0, -5, head_width=1, head_length=2, fc='k', ec='k', lw=0.5)
ax.arrow(t_em_6 // 24 + 1, 90, 0, -5, head_width=1, head_length=2, fc='k', ec='k', lw=0.5)
ax.arrow(t_em_7 // 24 + 1, 90, 0, -5, head_width=1, head_length=2, fc='k', ec='k', lw=0.5)
ax.arrow(t_em_8 // 24 + 1, 90, 0, -5, head_width=1, head_length=2, fc='k', ec='k', lw=0.5)
ax.arrow(t_em_9 // 24 + 1, 90, 0, -5, head_width=1, head_length=2, fc='k', ec='k', lw=0.5)

# letters
ax.text(0.08, 0.9, 'A', ha='center', va='center', size=9, transform=ax.transAxes)
ax1.text(0.08, 0.9, 'B', ha='center', va='center', size=9, transform=ax1.transAxes)
ax2.text(0.08, 0.85, 'C', ha='center', va='center', size=9, transform=ax2.transAxes)

plt.savefig(os.path.join(scenario_graphs_dirpath, 'C_source_sink_bis.PNG'), dpi=600, format='PNG', bbox_inches='tight')
plt.close()

## ----------- C usages cumulated vs. photosynthesis - shoot respiration

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(C_usages_days.day, (C_usages_days.Structure_shoot + C_usages_days.NS_shoot) / (C_usages_days.C_produced - C_usages_days.Respi_shoot) * 100,
        label=u'Shoot', color='g')
ax.plot(C_usages_days.day, (C_usages_days.Structure_roots + C_usages_days.NS_roots) / (C_usages_days.C_produced - C_usages_days.Respi_shoot) * 100,
        label=u'Roots', color='r')
ax.plot(C_usages_days.day, (C_usages_days.Respi_roots) / (C_usages_days.C_produced - C_usages_days.Respi_shoot) * 100, label=u'C loss by root respiration', color='b')
ax.plot(C_usages_days.day, C_usages_days.exudation / (C_usages_days.C_produced - C_usages_days.Respi_shoot) * 100, label=u'C loss by exudation', color='c')

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)

ax.set_xlabel(x_label_days)
ax.set_ylabel(u'Carbon usages :\nPhotosynthesis - Shoot repiration (%)')
ax.set_ylim(bottom=0, top=100.)

plt.savefig(os.path.join(scenario_graphs_dirpath, 'C_usages_cumulated_bis2.PNG'), dpi=600, format='PNG', bbox_inches='tight')
plt.close()

## ----------- SLN

fig = plt.figure(figsize=(4, 3))
# set height ratios for sublots
gs = gridspec.GridSpec(1, 1)

ax1 = plt.subplot(gs[0])
for i in range(1, 10):
    tmp = df_elt[(df_elt.element == 'LeafElement1') & (df_elt.metamer == i) & (df_elt.is_growing == 0)]  #
    tmp2 = tmp.merge(out_elt[['element', 'metamer', 't', 'senesced_length_element']], on=['element', 'metamer', 't'], how='left')
    SLA = tmp2[tmp2.senesced_length_element == 0].groupby(['day']).agg({'SLA': 'mean', 'SLN': 'mean'})
    SLA = SLA.merge(out_sam_days[['day', 'sum_TT', 'TT']], on='day').copy()
    ax1.plot(SLA.sum_TT, SLA.SLN, linestyle='-', color=colors[i], label=i)
ax1.get_yaxis().set_label_coords(-0.08, 0.5)
ax1.set_ylabel(u'Surfacic Leaf Nitrogen (g m$^{-2}$)')
ax1.set_ylim(0., 3.)
# x-axis
plt.setp(ax0.get_xticklabels(), visible=False)
ax1.set_xlabel(x_label_TT)
ax1.set_xlim(0, 700)

# put lened on first subplot
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)

# remove vertical gap between subplots
plt.subplots_adjust(hspace=.0)

plt.savefig(os.path.join(scenario_graphs_dirpath, 'SLN.PNG'), format='PNG', bbox_inches='tight', dpi=600)
plt.close()

## ----------- SLA and  Lamina area
fig = plt.figure(figsize=(4, 6))
# set height ratios for sublots
gs = gridspec.GridSpec(2, 1)

ax0 = plt.subplot(gs[0])
for i in range(1, 10):
    tmp = df_elt[(df_elt.element == 'LeafElement1') & (df_elt.metamer == i)]
    green_area = tmp.merge(out_sam[['t', 'sum_TT']], on='t').copy()
    ax0.plot(green_area.sum_TT, green_area.green_area * 10000, linestyle='-', color=colors[i], label=i)

## Formatting
ax0.get_yaxis().set_label_coords(-0.08, 0.5)
ax0.set_ylabel(u'Exposed lamina area (cm$^{2}$)')
ax0.set_xlabel(x_label_TT)
ax0.set_xlim(0, 700)
plt.setp(ax0.get_xticklabels(), visible=False)

ax2 = plt.subplot(gs[1], sharex=ax0)
for i in range(1, 10):
    tmp = df_elt[(df_elt.element == 'LeafElement1') & (df_elt.metamer == i) & (df_elt.is_growing == 0)]  #
    tmp2 = tmp.merge(out_elt[['element', 'metamer', 't', 'senesced_length_element']], on=['element', 'metamer', 't'], how='left')
    SLA = tmp2[tmp2.senesced_length_element == 0].groupby(['day']).agg({'SLA': 'mean', 'SLN': 'mean'})
    SLA = SLA.merge(out_sam_days[['day', 'sum_TT', 'TT']], on='day').copy()
    ax2.plot(SLA.sum_TT, SLA.SLA, linestyle='-', color=colors[i], label=i)
ax2.set_xlim(0, 700)
ax2.get_yaxis().set_label_coords(-0.08, 0.5)
ax2.set_ylabel(u'Specific Leaf Area (m$^{2}$ kg$^{-1}$)')
ax2.set_ylim(20., 50.)

yticks = ax2.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

# x-axis
ax2.set_xlabel(x_label_TT)

## Formatting
plt.subplots_adjust(hspace=.0)
ax0.legend(loc='upper left', bbox_to_anchor=(1, 0.35), frameon=True)

# letters
ax0.text(0.08, 0.9, 'A', ha='center', va='center', size=9, transform=ax0.transAxes)
ax2.text(0.08, 0.9, 'B', ha='center', va='center', size=9, transform=ax2.transAxes)

plt.savefig(os.path.join(scenario_graphs_dirpath, 'Dynamic_laminae2.PNG'), dpi=600, format='PNG', bbox_inches='tight')
plt.close()

## ---------- RUE

# We use PARa by element but green_area from MS (NA for tillers)
df_elt_all['cohort'] = df_elt_all.metamer
for i in range(0, df_elt_all.shape[0]):
    try:
        tiller_rank = int(df_elt_all.at[i, 'axis'][1:])
    except:
        continue
    df_elt_all.at[i, 'cohort'] += tiller_rank + 2

df_elt_all.drop(columns=['green_area'])
titi = df_elt_all[['t', 'organ', 'element', 'cohort', 'PARa']].merge(df_elt[['t', 'organ', 'element', 'metamer', 'green_area']],
                                                                     left_on=['t', 'organ', 'element', 'cohort'], right_on=['t', 'organ', 'element', 'metamer'], how='outer')
titi['RGa_MJ'] = titi['PARa'] * titi['green_area'] * 3600 / 2.02 * 10 ** -6
titi['PARa_MJ'] = titi['PARa'] * titi['green_area'] * 3600 / 4.6 * 10 ** -6
titi['PARa_pl'] = titi['PARa'] * titi['green_area'] * 3600
titi['day'] = titi['t'] // 24 + 1
RGa2 = titi.groupby(['day'], as_index=False).agg({'RGa_MJ': 'sum', 'PARa_MJ': 'sum', 'PARa_pl': 'sum'})
RGa2_cum = np.cumsum(RGa2.RGa_MJ)
PARa2_cum = np.cumsum(RGa2.PARa_MJ)

# Simplification
df_elt['RGa_MJ'] = df_elt['PARa'] * df_elt['green_area'] * df_elt['nb_replications'] * 3600 / 2.02 * 10 ** -6  # Il faudrait idealement utiliser les calculcs green_area et PARa des talles
RGa = df_elt.groupby(['day'])['RGa_MJ'].agg('sum')
RGa_cum = np.cumsum(RGa)
days = df_elt['day'].unique()

# Beer-Lambert
df_elt['green_area_rep'] = df_elt.green_area * df_elt.nb_replications
df_LAI = df_elt[(df_elt.element == 'LeafElement1')].groupby(['t']).agg({'green_area_rep': 'sum'})
df_LAI['LAI'] = df_LAI.green_area_rep * PLANT_DENSITY[1]
df_LAI['t'] = df_LAI.index
df_LAI['day'] = df_LAI['t'] // 24 + 1

toto = meteo[['t', 'PARi_MA4']].merge(df_LAI[['t', 'LAI', 'day']], on='t', how='inner')
toto['PARi_caribu'] = toto.PARi_MA4
ts_caribu = range(0, toto.shape[0], 4)
save = toto.at[0, 'PARi_MA4']
for i in range(0, toto.shape[0]):
    if i in ts_caribu:
        save = toto.at[i, 'PARi_MA4']
    toto.at[i, 'PARi_caribu'] = save

toto['PARint_BL'] = toto.PARi_caribu * (1 - np.exp(-0.4 * toto.LAI))
toto['RGint_BL_MJ'] = toto['PARint_BL'] * 3600 / 2.02 * 10 ** -6
toto['PARint_BL_MJ'] = toto['PARint_BL'] * 3600 / 4.6 * 10 ** -6
RGint_BL = toto.groupby(['day'])['RGint_BL_MJ'].agg('sum')
RGint_BL_cum = np.cumsum(RGint_BL)
PARint_BL_MJ = toto.groupby(['day'])['PARint_BL_MJ'].agg('sum')
PARint_BL_MJ_cum = np.cumsum(PARint_BL_MJ)

sum_dry_mass_shoot = df_axe.groupby(['day'], as_index=False)['sum_dry_mass_shoot'].agg('max')
sum_dry_mass_shoot_couvert = sum_dry_mass_shoot * PLANT_DENSITY[1]
sum_dry_mass = df_axe.groupby(['day'], as_index=False)['sum_dry_mass'].agg('max')
sum_dry_mass_couvert = sum_dry_mass * PLANT_DENSITY[1]

RUE_shoot = np.polyfit(RGa_cum, sum_dry_mass_shoot.sum_dry_mass_shoot, 1)[0]
RUE_shoot2 = np.polyfit(RGa2_cum, sum_dry_mass_shoot.sum_dry_mass_shoot, 1)[0]
RUE_shoot3 = np.polyfit(RGint_BL_cum, sum_dry_mass_shoot_couvert.sum_dry_mass_shoot, 1)[0]
RUE_shoot2_PAR = np.polyfit(PARa2_cum, sum_dry_mass_shoot.sum_dry_mass_shoot, 1)[0]
RUE_shoot3_PAR = np.polyfit(PARint_BL_MJ_cum, sum_dry_mass_shoot_couvert.sum_dry_mass_shoot, 1)[0]
RUE_plant = np.polyfit(RGa_cum, sum_dry_mass, 1)[0]
RUE_plant2 = np.polyfit(RGa2_cum, sum_dry_mass, 1)[0]

## Plot A
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(PARa2_cum, sum_dry_mass_shoot.sum_dry_mass_shoot, label='Shoot', color='g')
ax.plot(PARa2_cum, sum_dry_mass.sum_dry_mass, label='Shoot + Roots', color='k')

ax.text(0.08, 0.9, 'A', ha='center', va='center', size=9, transform=ax.transAxes)

## Formatting
# ax.legend(loc='upper left')
ax.set_xlabel('Cumulative absorbed PAR (MJ)')
ax.set_ylabel('Dry mass (g)')
plt.savefig(os.path.join(scenario_graphs_dirpath, 'RUE.PNG'), format='PNG', bbox_inches='tight', dpi=600)
plt.close()

## ---------- Graphe de RUE weekly
sum_dry_mass = df_axe.groupby(['day'], as_index=False)['sum_dry_mass'].agg('max')
tmp_prec7 = sum_dry_mass.copy()
tmp_prec7['day_prec7'] = tmp_prec7.day + 7
tmp_prec7['sum_dry_mass_prec7'] = tmp_prec7.sum_dry_mass
tmp = tmp_prec7[['day_prec7', 'sum_dry_mass_prec7']]
inc_dry_mass7 = sum_dry_mass.merge(tmp, left_on='day', right_on='day_prec7', how='left')
inc_dry_mass7['inc_dry_mass7'] = inc_dry_mass7.sum_dry_mass - inc_dry_mass7.sum_dry_mass_prec7
RUE_plant_week = inc_dry_mass7.merge(RGa2, on='day')
# RUE_plant_week['RGa_MJ_7'] = RUE_plant_week.RGa_MJ.rolling(7).sum()
# RUE_plant_week['RGint_BL_7'] = RGint_BL.rolling(7).sum()
RUE_plant_week['PARa_MJ_7'] = RUE_plant_week.PARa_MJ.rolling(7).sum()
# RUE_plant_week['PARa_pl_7'] = RUE_plant_week.PARa_pl.rolling(7).sum()
# RUE_plant_week['PARint_BL_7'] = PARint_BL_MJ.rolling(7).sum()
# RUE_plant_week['RUE7'] = RUE_plant_week.inc_dry_mass7 / RUE_plant_week.RGa_MJ_7
# RUE_plant_week['RUE_BL7'] = RUE_plant_week.inc_dry_mass7  * PLANT_DENSITY[1] / RUE_plant_week.RGint_BL_7
RUE_plant_week['RUE7_PAR'] = RUE_plant_week.inc_dry_mass7 / RUE_plant_week.PARa_MJ_7
# RUE_plant_week['RUE_BL7_PAR'] = RUE_plant_week.inc_dry_mass7  * PLANT_DENSITY[1] / RUE_plant_week.PARint_BL_7
RUE_plant_week = RUE_plant_week.merge(out_sam_days, on='day').copy()

sum_dry_mass_shoot = df_axe.groupby(['day'], as_index=False)['sum_dry_mass_shoot'].agg('max')
tmp_prec7 = sum_dry_mass_shoot.copy()
tmp_prec7['day_prec7'] = tmp_prec7.day + 7
tmp_prec7['sum_dry_mass_prec7'] = tmp_prec7.sum_dry_mass_shoot
tmp = tmp_prec7[['day_prec7', 'sum_dry_mass_prec7']]
inc_dry_mass7 = sum_dry_mass_shoot.merge(tmp, right_on='day_prec7', left_on='day', how='left')
inc_dry_mass7['inc_dry_mass7_shoot'] = inc_dry_mass7.sum_dry_mass_shoot - inc_dry_mass7.sum_dry_mass_prec7
RUE_shoot_week = inc_dry_mass7.merge(RGa2, on='day')
# RUE_shoot_week['RGa_MJ_7'] = RUE_shoot_week.RGa_MJ.rolling(7).sum()
# RUE_shoot_week['RGint_BL_7'] = RGint_BL.rolling(7).sum()
RUE_shoot_week['PARa_MJ_7'] = RUE_shoot_week.PARa_MJ.rolling(7).sum()
# RUE_shoot_week['PARa_pl_7'] = RUE_shoot_week.PARa_pl.rolling(7).sum()
# RUE_shoot_week['PARint_BL_7'] = PARint_BL_MJ.rolling(7).sum()
# RUE_shoot_week['RUE7'] = RUE_shoot_week.inc_dry_mass7_shoot / RUE_shoot_week.RGa_MJ_7
# RUE_shoot_week['RUE_BL7'] = RUE_shoot_week.inc_dry_mass7_shoot  * PLANT_DENSITY[1] / RUE_shoot_week.RGint_BL_7
RUE_shoot_week['RUE7_PAR'] = RUE_shoot_week.inc_dry_mass7_shoot / RUE_shoot_week.PARa_MJ_7
# RUE_shoot_week['RUE_BL7_PAR'] = RUE_shoot_week.inc_dry_mass7_shoot  * PLANT_DENSITY[1] / RUE_shoot_week.PARint_BL_7
RUE_shoot_week = RUE_shoot_week.merge(out_sam_days, on='day').copy()

## Plot
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(RUE_plant_week.sum_TT, RUE_plant_week.RUE7_PAR, label='Shoot + Roots', color='k', marker='o', linestyle='', markersize=3, mfc='none')
ax.plot(RUE_shoot_week.sum_TT, RUE_shoot_week.RUE7_PAR, label='Shoot', color='g', marker='o', linestyle='', markersize=3)

# second x-axis
ax30 = ax.twiny()
new_tick_locations = np.array([3, 4, 5, 6, 7, 8])
old_tick_locations = np.array([69.6, 164.6, 271.71, 364.8, 465.7, 555])  # df_HS_day.groupby(['nb_lig']).agg({'sum_TT':'min'})

ax30.set_xlim(ax.get_xlim())
ax30.set_xticks(old_tick_locations)
ax30.set_xticklabels(new_tick_locations)
ax30.set_xlabel('Number of ligulated leaves')
ax30.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
ax30.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
ax30.spines['bottom'].set_position(('outward', 36))

ax.text(0.1, 0.9, 'B', ha='center', va='center', size=9, transform=ax.transAxes)

## Formatting
ax.set_xlabel(x_label_TT)
ax.set_xlim(0, 700)
ax.set_ylim(0, 5)
ax.set_ylabel(u'Weekly Radiation Use Efficiency (g MJ$^{-1}$)')
ax.legend(loc='center left', frameon=True, numpoints=1, bbox_to_anchor=(1, 0.5))
plt.savefig(os.path.join(scenario_graphs_dirpath, 'RUE_PAR_weekly.PNG'), format='PNG', bbox_inches='tight', dpi=600)
plt.close()

## --- Join 2 RUE graphs

# fig, axs = plt.subplots(1,2, figsize = (8,3) )
#
# image = mpimg.imread(os.path.join(scenario_graphs_dirpath, 'RUE.PNG'))
# axs[0].imshow(image)
# axs[0].axis('off')
#
# image = mpimg.imread(os.path.join(scenario_graphs_dirpath, 'RUE_PAR_weekly.PNG'))
# axs[1].imshow(image)
# axs[1].axis('off')
#
# plt.tight_layout(pad=0.05)
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.subplots_adjust(top=0.95)
# plt.savefig( os.path.join(scenario_graphs_dirpath,'RUE_2graphs.PNG'), format='PNG', bbox_inches='tight', dpi=600)
# plt.close()


## ---------- RUE vs. PTQ

df_corr = pd.DataFrame({'PTQ': meteo_days.PTQ2_smooth,
                        'RUE_plant': RUE_plant_week.RUE7_PAR,
                        'RUE_shoot': RUE_shoot_week.RUE7_PAR,
                        })

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(df_corr.PTQ, df_corr.RUE_plant, color='k', marker='o', linestyle='', markersize=3, mfc='none')
ax.plot(df_corr.PTQ, df_corr.RUE_shoot, color='g', marker='o', linestyle='', markersize=3)

## Formatting
ax.set_ylim(0, 5)
ax.set_xlim(0, 0.4)
ax.set_ylabel(u'Weekly Radiation Use Efficiency (g MJ$^{-1}$)')
ax.set_xlabel(u'Photothermal quotient (mol m$^{-2}$ d$^{-1}$ at 12°C)')
plt.savefig(os.path.join(scenario_graphs_dirpath, 'RUE_both_vs_PTQ.PNG'), format='PNG', bbox_inches='tight', dpi=600)
plt.close()

## ----------  GLN / GAI / fermeture couvert / surfacic PARa

## -- Surfaces senescentes
leaf_ligulation = df_elt[(df_elt.element == 'LeafElement1') & (df_elt.is_growing == 0)].groupby('metamer').agg({'t': 'min'})
leaf_ligulation = leaf_ligulation.t.to_dict()

df_elt_max_ga = df_elt[(df_elt.element == 'LeafElement1')].groupby(['metamer']).agg({'green_area_rep': 'max', 'green_area': 'max'})
df_elt_max_ga['green_area_rep_max'] = df_elt_max_ga.green_area_rep
df_elt_max_ga['green_area_max'] = df_elt_max_ga.green_area
df_elt_max_ga['metamer'] = df_elt_max_ga.index

df_senesc = df_elt[(df_elt.element == 'LeafElement1')].merge(df_elt_max_ga[['green_area_rep_max', 'green_area_max', 'metamer']], left_on='metamer', right_on='metamer').copy()
df_senesc['senesc_area_rep'] = 0.
df_senesc['senesc_area'] = 0.
for i in df_senesc.index:
    if df_senesc.at[i, 't'] > leaf_ligulation.get(df_senesc.at[i, 'metamer'], 5000):
        df_senesc.at[i, 'senesc_area_rep'] = df_senesc.at[i, 'green_area_rep_max'] - df_senesc.at[i, 'green_area_rep']
        df_senesc.at[i, 'senesc_area'] = df_senesc.at[i, 'green_area_max'] - df_senesc.at[i, 'green_area']

tmp = expand_grid({'t': df_senesc.t.drop_duplicates(), 'metamer': df_senesc.metamer.drop_duplicates()})
tmp2 = tmp.merge(df_elt_max_ga[['metamer', 'green_area_rep_max', 'green_area_max']], left_on=['metamer'], right_on=['metamer'], how='left')
tmp2['fully_senesc_area_rep'] = tmp2.green_area_rep_max
tmp2['fully_senesc_area'] = tmp2.green_area_max

tmp3 = tmp2[['t', 'metamer', 'fully_senesc_area', 'fully_senesc_area_rep']].merge(df_senesc, left_on=['t', 'metamer'], right_on=['t', 'metamer'], how='left')
tmp3['senesc_area_all'] = 0.
tmp3['senesc_area_all_rep'] = 0.
for i in tmp3.index:
    if tmp3.at[i, 't'] < leaf_emergence.get((1, tmp3.at[i, 'metamer']), 0):
        tmp3.at[i, 'senesc_area_all_rep'] = 0
        tmp3.at[i, 'senesc_area_all'] = 0
    else:
        if np.isnan(tmp3['senesc_area_rep'][i]):
            tmp3.at[i, 'senesc_area_all_rep'] = tmp3.at[i, 'fully_senesc_area_rep']
            tmp3.at[i, 'senesc_area_all'] = tmp3.at[i, 'fully_senesc_area']
        else:
            tmp3.at[i, 'senesc_area_all_rep'] = tmp3.at[i, 'senesc_area_rep']
            tmp3.at[i, 'senesc_area_all'] = tmp3.at[i, 'senesc_area']

# Pourcentage de vert : FAUX pour les feuilles qui ne se ligulent pas au cours de la simulation
tmp3['Pge_green_MS'] = tmp3.green_area / tmp3.green_area_max
tmp3['Pge_green_rep'] = tmp3.green_area_rep / tmp3.green_area_rep_max

df_senesc_tot = tmp3.groupby(['t']).agg({'senesc_area_all_rep': 'sum', 'Pge_green_MS': 'sum'})
df_senesc_tot['t'] = df_senesc_tot.index

df_LAI = df_LAI.merge(df_senesc_tot, left_on='t', right_on='t').copy()
df_LAI['LAI_all'] = df_LAI.LAI + df_LAI.senesc_area_all_rep * PLANT_DENSITY[plant]

df_LAI['day'] = df_LAI.t // 24 + 1
LAI = df_LAI.groupby(['day']).agg({'LAI': 'mean'})
LAI_all = df_LAI.groupby(['day']).agg({'LAI_all': 'mean'})

LAI = LAI.merge(out_sam_days[['day', 'sum_TT', 'TT']], on='day').copy()
LAI_all = LAI_all.merge(out_sam_days[['day', 'sum_TT', 'TT']], on='day').copy()

## -- GAI

## pb de gaines : lorsque gaine 1 est sénescente, gaine 2 est toujours majoritairement cachée
## pour le GAI, la surface considérée = 50% surface du cylindre

tmp = df_elt[(df_elt.organ == 'sheath')].groupby(['t', 'metamer'], as_index=False).agg({'green_area_rep': 'sum'})
S_gaines = tmp.groupby(['t'], as_index=False).agg({'green_area_rep': 'sum'})
S_gaines['S_GAI'] = S_gaines.green_area_rep / 2
S_gaines['day'] = S_gaines.t // 24 + 1
S_gaines_days = S_gaines.groupby(['day'], as_index=False).agg({'S_GAI': 'mean'})
GAI = LAI.merge(S_gaines_days, on='day')
GAI['GAI'] = GAI.LAI + GAI.S_GAI

TT_Mariem9 = [570.79, 593.26, 719.1, 768.54, 1006.74, 1020.22]
GAI_Mariem9 = [0.4127, 0.5137, 0.6687, 0.7377, 3.9999, 4.2039]

TT_Mariem10 = [560.15, 587.45, 709.55, 763.86, 1024.72, 1024.82, 1025.18, 1297.87, 1302.11, 1306.58, 1640.39, 1649.14, 1649.73]
GAI_Mariem10 = [0.3575, 0.484, 0.6576, 0.7794, 3.9037, 3.9049, 4.3055, 5.9101, 5.9272, 5.6305, 5.3831, 5.3654, 5.3641]

TT_Mariem8 = [265.72, 335.37, 387.63, 439.85, 518.14, 631.16, 835.52, 900.7, 1022.51, 1104.92, 1405]
GAI_Mariem8 = [0.1457, 0.2526, 0.3064, 0.4307, 0.6964, 1.1917, 1.9883, 2.3066, 2.6788, 3.3675, 4.4124]

## -- GLN
GLN_Mariem_TT = [246, 299, 357, 383, 456, 498, 559, 704, 792, 846, 903, 1010, 1083, 1186, 1290, 1378, 1469, 1565, 1672, 1806, 1947]
GLN_Mariem_GLN = [1.1, 1.7, 1.9, 2.8, 3.8, 4.3, 5, 3.8, 3.7, 3.8, 3.4, 3.8, 4.3, 4, 3.8, 3.3, 2.9, 2.8, 2.3, 1.7, 0.9]

GLN_Mariem_TT2 = [238, 295, 341, 451, 485, 561, 705, 796, 834, 890, 1008, 1076, 1186, 1289, 1380, 1466, 1602, 1712, 1871, 1958]
GLN_Mariem_GLN2 = [1.1, 1.8, 2.6, 3.6, 4.4, 5, 4.7, 3.9, 3.7, 3.9, 3.8, 4.4, 5, 5, 4.8, 4.1, 3.6, 2.9, 2.4, 2.1]

GLN_Mariem_TT3 = [154, 204, 250, 258, 292, 380, 411, 442, 488, 534, 576, 603, 667, 727, 810, 989, 1065, 1275, 1362, 1524, 1835]
GLN_Mariem_GLN3 = [0.3, 0.7, 1.2, 1.4, 1.7, 2.5, 2.7, 3.1, 3.5, 3.8, 4.3, 4.4, 4.3, 4.2, 3.5, 4, 4.5, 5.6, 5.3, 4.7, 2.5]

GLN_Mariem_TT4 = [393, 488, 548, 598, 659, 700, 829, 898, 970, 1038, 1122, 1372, 1440, 1524, 1782, 1972, 2192]
GLN_Mariem_GLN4 = [2.8, 3.7, 4.3, 4.5, 5.4, 5, 3.6, 3.8, 4.5, 5.2, 5.4, 5.5, 5.6, 5.4, 4.5, 2.5, 0]

df_senesc_tot = df_senesc_tot.merge(out_sam[['t', 'sum_TT']], left_on='t', right_on='t').copy()
tmp = df_senesc_tot[(df_senesc_tot.t < leaf_emergence[(1, 9)]) & (df_senesc_tot.Pge_green_MS != 0.)].copy()  # only valid before emergence of lamina 9 because we don't knpw its final length

df_gln = tmp

## -- Part rayonnement absrobé par les surfaces vertes de la plante
titi['PARa_surface'] = titi.PARa * titi.green_area * 250
titi['PARa_surface2'] = titi.PARa * titi.green_area
titi_caribu = titi[(titi.t % 4 == 0) & (titi['element'].isin(['StemElement', 'LeafElement1']))]
tutu = titi_caribu.groupby(['t'], as_index=False).agg({'PARa_surface': 'sum',
                                                       'green_area': 'sum'})
tutu = tutu.merge(meteo, on='t').copy()
tutu_days = tutu.groupby(['day'], as_index=False).agg({'PARa_surface': 'sum',
                                                       'PARi': 'sum',
                                                       'PARi_MA4': 'sum',
                                                       'green_area': 'mean'})
tutu_days = tutu_days.merge(out_sam_days, on='day').copy()

tutu_days['ratio_PARa_PARi'] = tutu_days.PARa_surface / tutu_days.PARi_MA4

tmp = titi[titi['element'].isin(['StemElement', 'LeafElement1'])]
tutu2 = tmp.groupby(['t'], as_index=False).agg({'PARa_surface2': 'sum',
                                                'green_area': 'sum'})
tutu2 = tutu2.merge(meteo, on='t').copy()
tutu2_days = tutu2.groupby(['day'], as_index=False).agg({'PARa_surface2': 'sum',
                                                         'PARi': 'sum',
                                                         'PARi_MA4': 'sum',
                                                         'green_area': 'mean'})
tutu2_days = tutu2_days.merge(out_sam_days, on='day').copy()

tutu2_days['PARa_surfacique'] = tutu2_days.PARa_surface2 / tutu2_days.green_area
tutu2_days['PARa_mol_m2_d'] = tutu2_days['PARa_surfacique'] * 3600 * 10 ** -6

## -- Graph

fig = plt.figure(figsize=(4, 12))
# set height ratios for sublots
gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1])

# the fisrt subplot
ax0 = plt.subplot(gs[0])
ax0.plot(df_gln.sum_TT, df_gln.Pge_green_MS, linestyle='-', color='k')
ax0.plot(np.array(GLN_Mariem_TT) - TT_since_sowing, GLN_Mariem_GLN, marker='o', color='k', linestyle='')
ax0.plot(np.array(GLN_Mariem_TT2) - TT_since_sowing, GLN_Mariem_GLN2, marker='o', mfc='none', color='k', linestyle='')
ax0.plot(np.array(GLN_Mariem_TT3) - TT_since_sowing, GLN_Mariem_GLN3, marker='D', color='k', linestyle='')
ax0.plot(np.array(GLN_Mariem_TT4) - TT_since_sowing, GLN_Mariem_GLN4, marker='^', color='k', linestyle='')
ax0.set_xlim(0, 700)
ax0.set_ylim(bottom=0, top=6)

# align y axis label
ax0.get_yaxis().set_label_coords(-0.08, 0.5)
ax0.set_ylabel(u'Green Leaf Number')

# the second subplot
ax1 = plt.subplot(gs[1], sharex=ax0)
ax1.plot(GAI.sum_TT, GAI.GAI, linestyle='-', color='k')
ax1.plot(np.array(TT_Mariem8) - TT_since_sowing, GAI_Mariem8, marker='^', color='k', linestyle='')
ax1.plot(np.array(TT_Mariem9) - TT_since_sowing, GAI_Mariem9, marker='o', color='k', linestyle='')
ax1.plot(np.array(TT_Mariem10) - TT_since_sowing, GAI_Mariem10, marker='o', mfc='none', color='k', linestyle='')
ax1.set_xlim(0, 700)
ax1.set_ylim(0, 5)

# align y axis label
ax1.get_yaxis().set_label_coords(-0.08, 0.5)
ax1.set_ylabel('Global Area Index')
# remove last tick label for the second subplot
yticks = ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

## third subplot
ax2 = plt.subplot(gs[2], sharex=ax0)
ax2.plot(tutu_days.sum_TT, tutu_days.ratio_PARa_PARi, color='k')
ax2.set_xlim(0, 700)
ax2.set_ylim(0, 1)

ax2.get_yaxis().set_label_coords(-0.08, 0.5)
ax2.set_ylabel(u'Faction of incident PAR\nabsorbed by the canopy')
yticks = ax2.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

## fourth subplot
ax3 = plt.subplot(gs[3], sharex=ax0)
ax3.plot(tutu2_days.sum_TT, tutu2_days.PARa_mol_m2_d, color='k')
ax3.set_xlim(0, 700)

ax3.get_yaxis().set_label_coords(-0.08, 0.5)
ax3.set_ylabel(u'Surfacic PAR absorbed (mol m$^{-2}$ d$^{-1}$)')
yticks = ax3.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

# second x-axis
ax30 = ax3.twiny()
new_tick_locations = np.array([3, 4, 5, 6, 7, 8])
old_tick_locations = np.array([69.6, 164.6, 271.71, 364.8, 465.7, 555])  # df_HS_day.groupby(['nb_lig']).agg({'sum_TT':'min'})

ax30.set_xlim(ax3.get_xlim())
ax30.set_xticks(old_tick_locations)
ax30.set_xticklabels(new_tick_locations)
ax30.set_xlabel('Number of ligulated leaves')
ax30.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
ax30.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
ax30.spines['bottom'].set_position(('outward', 36))

## Formatting
# shared axis X
plt.setp(ax0.get_xticklabels(), visible=False)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
# remove vertical gap between subplots
plt.subplots_adjust(hspace=.0)
ax3.set_xlabel(u'Time since leaf 4 emergence (°Cd)')

ax0.text(0.08, 0.9, 'A', ha='center', va='center', size=9, transform=ax0.transAxes)
ax1.text(0.08, 0.9, 'B', ha='center', va='center', size=9, transform=ax1.transAxes)
ax2.text(0.08, 0.9, 'C', ha='center', va='center', size=9, transform=ax2.transAxes)
ax3.text(0.08, 0.9, 'D', ha='center', va='center', size=9, transform=ax3.transAxes)

plt.savefig(os.path.join(scenario_graphs_dirpath, 'Couvert.PNG'), format='PNG', bbox_inches='tight', dpi=600)
plt.close()

## ---------- Shoot/ Roots
df_axe['sum_dry_mass_phloem_shoot'] = df_axe.dry_mass_phloem * shoot_plant_mstruct_ratio
df_axe['sum_dry_mass_shoot_MS'] = df_axe.sum_dry_mass_phloem_shoot + \
                                  df_hz.groupby(['t', 'plant', 'axis'])['sum_dry_mass'].agg('sum').values + \
                                  df_elt.groupby(['t', 'plant', 'axis'])['sum_dry_mass'].agg('sum').values

shoot_roots_days = df_axe.groupby(['day']).agg({'shoot_roots_ratio': 'mean',
                                                'sum_dry_mass_shoot': 'mean',
                                                'sum_dry_mass_shoot_MS': 'mean',
                                                'sum_dry_mass_roots': 'mean'})
shoot_roots_days = shoot_roots_days.merge(out_sam_days, on='day').copy()

## Plot
fig, (ax1) = plt.subplots(1, figsize=(4, 3))

ax1.plot(shoot_roots_days.sum_TT, shoot_roots_days.sum_dry_mass_shoot * 250 * 10 ** -2, color='g')  # convert from g.plant-1 to t.ha-1
ax1.plot(shoot_roots_days.sum_TT, shoot_roots_days.sum_dry_mass_shoot_MS * 250 * 10 ** -2, color='g', linestyle=':')
ax1.plot(shoot_roots_days.sum_TT, shoot_roots_days.sum_dry_mass_roots * 250 * 10 ** -2, color='r')

ax00 = ax1.twinx()
ax00.set_ylim(0, 3)
ax00.plot(shoot_roots_days.sum_TT, shoot_roots_days.shoot_roots_ratio, color='k')

ax00.get_yaxis().set_label_coords(1.1, 0.5)
ax00.set_ylabel('Shoot:Roots dry mass ratio')

# second x-axis
ax30 = ax1.twiny()
new_tick_locations = np.array([3, 4, 5, 6, 7, 8])
old_tick_locations = np.array([69.6, 164.6, 271.71, 364.8, 465.7, 555])  # df_HS_day.groupby(['nb_lig']).agg({'sum_TT':'min'})

ax30.set_xlim(ax1.get_xlim())
ax30.set_xticks(old_tick_locations)
ax30.set_xticklabels(new_tick_locations)
ax30.set_xlabel('Number of ligulated leaves')
ax30.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
ax30.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
ax30.spines['bottom'].set_position(('outward', 36))

## Formatting
ax1.set_ylabel(u'Dry mass (t ha$^{-1}$)')
ax1.set_xlabel(x_label_TT)
ax1.set_xlim(0, 700)
ax1.set_ylim(bottom=0.)
plt.savefig(os.path.join(scenario_graphs_dirpath, 'Shoot_roots_TT_tha.PNG'), format='PNG', bbox_inches='tight', dpi=600)
plt.close()

## ----------- N content

dt_RB_N = [1.2, 1.2, 1.7, 1.0]
dt_RB_N_confint = [0.11, 0.12, 0.19, 0.10]
dt_RB_TT = [10, 260, 330, 550]

dt_RB_Ns = [4.4, 3.7, 4.6, 3.3]
dt_RB_Ns_confint = [0.18, 0.35, 0.59, 0.19]
dt_RB_Ns_TT = [10, 260, 330, 550]

# dt_JB_Ns = [3.9]
# dt_JB_Ns_confint = [0.8]
# dt_JB_Ns_TT = [560 - TT_since_sowing]

df_hz['N_tot'] = (df_hz.N_content * df_hz.sum_dry_mass) / 100
df_axe['sum_mstruct_hz_elt_MS'] = (df_hz.groupby(['t', 'plant', 'axis'])['mstruct'].agg('sum').values + \
                                   df_elt.groupby(['t', 'plant', 'axis'])['mstruct'].agg('sum').values)
hz_elt_MS_plant_mstruct_ratio = df_axe['sum_mstruct_hz_elt_MS'] / df_axe.mstruct
df_phloem['Dry_Mass'] = ((df_phloem.sucrose * 1E-6 * 12) / 0.42 + \
                         (df_phloem.amino_acids * 1E-6 * 14) / 0.135)
df_axe['sum_dry_mass_shoot_MS'] = (df_phloem.Dry_Mass.reset_index(drop=True) * hz_elt_MS_plant_mstruct_ratio + \
                                   df_hz.groupby(['t', 'plant', 'axis'])['sum_dry_mass'].agg('sum').values + \
                                   df_elt.groupby(['t', 'plant', 'axis'])['sum_dry_mass'].agg('sum').values)
df_axe['N_content_shoot_MS'] = (df_phloem.N_tot.reset_index(drop=True) * hz_elt_MS_plant_mstruct_ratio + \
                                df_hz.groupby(['t', 'plant', 'axis'])['N_tot'].agg('sum').values + \
                                df_elt.groupby(['t', 'plant', 'axis'])['N_tot'].agg('sum').values) / df_axe['sum_dry_mass_shoot_MS'] * 100

N_shoot_days = df_axe.groupby(['day']).agg({'N_content_shoot': 'mean'})
N_roots_days = df_axe.groupby(['day']).agg({'N_content_roots': 'mean'})
N_shoot_MS_days = df_axe.groupby(['day']).agg({'N_content_shoot_MS': 'mean'})
N_shoot_days = N_shoot_days.merge(out_sam_days, on='day').copy()
N_roots_days = N_roots_days.merge(out_sam_days, on='day').copy()
N_shoot_MS_days = N_shoot_MS_days.merge(out_sam_days, on='day').copy()

## NNc
sum_dry_mass_shoot_days = df_axe.groupby(['day'])['sum_dry_mass_shoot'].mean()
DM_t_ha = sum_dry_mass_shoot_days * 250 * 10 ** -2  # convert from g.plant-1 to t.ha-1
N_content_critical = np.where(DM_t_ha < 1.55, 4.4, 5.35 * DM_t_ha ** -0.442)  # from Justes 1994 : valid at field scale from Feekes 3 i.e. mid tillering
ref_DM = np.arange(0, 3, 0.2)
ref_N_crit = np.where(ref_DM < 1.55, 4.4, 5.35 * ref_DM ** -0.442)  # from Justes 1994 : valid at field scale from Feekes 3 i.e. mid tillering

## Graph
fig = plt.figure(figsize=(4, 3))

ax2 = fig.add_subplot(111)
ax2.plot(N_shoot_days.sum_TT, N_shoot_days.N_content_shoot, color='g')
ax2.plot(N_roots_days.sum_TT, N_roots_days.N_content_roots, color='r')
ax2.errorbar(dt_RB_TT[:2], dt_RB_N[:2], yerr=dt_RB_N_confint[:2], marker='o', color='r', linestyle='')
ax2.errorbar(dt_RB_TT[2:], dt_RB_N[2:], yerr=dt_RB_N_confint[2:], marker='o', color='r', linestyle='', alpha=0.5)

ax2.plot(N_shoot_MS_days.sum_TT, N_shoot_MS_days.N_content_shoot_MS, color='g', linestyle=':')
ax2.errorbar(dt_RB_Ns_TT[:2], dt_RB_Ns[:2], yerr=dt_RB_Ns_confint[:2], marker='o', color='g', linestyle='')
ax2.errorbar(dt_RB_Ns_TT[2:], dt_RB_Ns[2:], yerr=dt_RB_Ns_confint[2:], marker='o', color='g', linestyle='', alpha=0.5)

ax2.set_xlabel(x_label_TT)
ax2.set_xlim(0, 700)
ax2.set_ylim(bottom=0., top=5.5)
ax2.set_ylabel('Nitrogen fraction (% dry mass)')

rect = [0.55, 0.35, 0.40, 0.30]
ax1 = add_subplot_axes(ax2, rect)
ax1.plot(DM_t_ha[N_shoot_days.day], N_shoot_days.N_content_shoot, color='g')
ax1.plot(ref_DM, ref_N_crit, color='k', linestyle='--')
ax1.set_xlabel('Shoot dry mass (t ha$^{-1}$)', size=8)
ax1.get_xaxis().set_label_coords(0.5, -0.2)
ax1.set_ylim(bottom=3., top=5.)
ax1.set_yticks([3, 4, 5])
ax1.set_xticks([0, 1, 2, 3])
ax1.xaxis.set_tick_params(labelsize=7)
ax1.yaxis.set_tick_params(labelsize=7)

# second x-axis
ax30 = ax2.twiny()
new_tick_locations = np.array([3, 4, 5, 6, 7, 8])
old_tick_locations = np.array([69.6, 164.6, 271.71, 364.8, 465.7, 555])  # df_HS_day.groupby(['nb_lig']).agg({'sum_TT':'min'})

ax30.set_xlim(ax2.get_xlim())
ax30.set_xticks(old_tick_locations)
ax30.set_xticklabels(new_tick_locations)
ax30.set_xlabel('Number of ligulated leaves')
ax30.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
ax30.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
ax30.spines['bottom'].set_position(('outward', 36))

# fertilizatrion time
ax2.axvline(x=416, linestyle=':', color='grey')
ax1.axvline(x=0.86, linestyle=':', color='grey')

plt.savefig(os.path.join(scenario_graphs_dirpath, 'N_content_inset_TT.PNG'), format='PNG', bbox_inches='tight', dpi=600)
plt.close()

## -----------  Conc phloem

df_phloem_day = df_phloem.groupby(['day'], as_index=False).agg({'sucrose': 'mean', 'amino_acids': 'mean',
                                                                'Conc_Sucrose': 'mean', 'Conc_Amino_Acids': 'mean'})
df_phloem_day['Conc_Sucrose_smooth'] = df_phloem_day.Conc_Sucrose.rolling(7).mean()
df_phloem_day['Conc_AA_smooth'] = df_phloem_day.Conc_Amino_Acids.rolling(7).mean()

## -- Graph
fig, (ax) = plt.subplots(1, figsize=(4, 3))
ax.plot(df_phloem_day.day, df_phloem_day.Conc_Sucrose, label=u'Sucrose (µmol)', color='b')
ax.plot(df_phloem_day.day, df_phloem_day.Conc_Amino_Acids, label=u'Amino acids (µmol N)', color='r')
ax.legend(loc='upper right', frameon=True)
ax.set_xlim(0, 120)
ax.set_ylim(0, 600)
ax.set_ylabel(u'$\it{Phloem}$ concentration (µmol g$^{-1}$)')
ax.set_xlabel('Days since leaf 4 emergence')
ax.set_xlim(0, 120)

# second x-axis
ax2 = ax.twiny()
new_tick_locations = np.array([0, 100, 200, 300, 400, 500, 600])
old_tick_locations = np.array([0, 17.8, 38.5, 63.5, 81, 93, 105])

ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(old_tick_locations)
ax2.set_xticklabels(new_tick_locations)
ax2.set_xlabel(u'Thermal time since leaf 4 emergence (°Cd)')
ax2.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
ax2.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
ax2.spines['bottom'].set_position(('outward', 36))

plt.savefig(os.path.join(scenario_graphs_dirpath, 'Conc_phloem_TT_d.PNG'), format='PNG', bbox_inches='tight', dpi=600)
plt.close()

## --------- Internode length

fig, (ax1) = plt.subplots(1, figsize=(4, 3))
for i in range(1, 10):
    tmp = df_hz[(df_hz.metamer == i)].merge(out_sam[['t', 'sum_TT']], on='t').copy()
    ax1.plot(tmp.sum_TT, tmp.internode_L * 100, linestyle='-', color=colors[i], label=i)

## Formatting
ax1.set_ylabel(u'Internode length (cm)')
ax1.set_xlabel(x_label_TT)
plt.savefig(os.path.join(scenario_graphs_dirpath, 'InternodeL_TT.PNG'), format='PNG', bbox_inches='tight', dpi=600)
plt.close()

# --- Correlations

df_corr['Conc_Sucrose_phloem_smooth'] = df_phloem_day['Conc_Sucrose_smooth']

mask = ~np.isnan(df_corr.PTQ) & ~np.isnan(df_corr.Conc_Sucrose_phloem_smooth)
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_corr.PTQ[mask], df_corr.Conc_Sucrose_phloem_smooth[mask])
scipy.stats.pearsonr(df_corr.PTQ[mask], df_corr.Conc_Sucrose_phloem_smooth[mask])

print u'\n Correlation Sucrose_phloem vs. PTQ :'
print 'R2 = ', r_value, ' p = ', p_value
print u'\n'

mask = ~np.isnan(df_corr.PTQ) & ~np.isnan(df_corr.RUE_shoot)
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_corr.PTQ[mask], df_corr.RUE_shoot[mask])
scipy.stats.pearsonr(df_corr.PTQ[mask], df_corr.RUE_shoot[mask])

print u'\n Correlation RUE shoot vs. PTQ :'
print 'R2 = ', r_value, ' p = ', p_value
print u'\n'
