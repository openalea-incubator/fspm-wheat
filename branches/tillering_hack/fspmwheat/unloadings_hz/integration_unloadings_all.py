# -*- coding: latin-1 -*-

import os
import pandas as pd
import numpy as np

all_files = os.listdir(os.getcwd())

files = [i for i in all_files if ((i[-4:] == '.txt') and (i[:3] == 'all')) ]

all_df_int = pd.DataFrame()

for i_file in files:
    if os.stat(i_file).st_size != 0:
        df = pd.read_csv(i_file, sep=',', header = None)
        df.columns = ['comp','metamer','t_int','unloading_sucrose','unloading_aa']

        map_elt = df[['comp','metamer']].drop_duplicates()
        for i_elt in map_elt.index:
            df_elt = df[(df.comp == map_elt.comp[i_elt]) & (df.metamer == map_elt.metamer[i_elt])].copy()
            df_elt.reset_index(drop=True, inplace=True)

            # delta_t_int
            df_elt['delta_t_int'] = 0.
            for i in df_elt.index:
                if df_elt.at[i,'t_int'] != 0.:
                    df_elt.at[i,'delta_t_int'] = df_elt.at[i,'t_int'] - df_elt.at[i-1,'t_int']

            # tmp variable
            df_elt['prod_sucrose'] = df_elt.unloading_sucrose * df_elt.delta_t_int
            df_elt['prod_aa'] = df_elt.unloading_aa * df_elt.delta_t_int

            # integration
            df_int = pd.DataFrame({ 'comp': map_elt.comp[i_elt],
                                    'metamer': map_elt.metamer[i_elt],
                                    't_sim': [int(i_file[4:-4])],
                                    'unloading_sucrose_int': [sum(df_elt.prod_sucrose)],
                                    'unloading_aa_int': [sum(df_elt.prod_aa)],
                                    'delta_t_int_sum': [sum(df_elt.delta_t_int)] })

            # concat to all_df
            all_df_int = all_df_int.append(df_int,  ignore_index=True)

    print i_file

all_df_int = all_df_int.sort_values(by = 't_sim')

output_file = 'all_integration_unloadings.csv'
all_df_int.to_csv(output_file, na_rep='NA', index=False)

## -- Add nb_replications
df = pd.read_csv('all_integration_unloadings.csv', sep=',')
ref = pd.read_csv('C:/Users/mngauthier/Documents/Marion_These/Modeles/fspm-wheat/trunk/fspmwheat/outputs/hiddenzones_states.csv')
ref = ref[ref.axis == 'MS'].copy()

repli = ref[['t','metamer','nb_replications']].drop_duplicates()
res = df.merge(repli, left_on = ['metamer','t_sim'], right_on = ['metamer','t'], how='left')
res.at[ np.isnan(res.nb_replications), 'nb_replications'] = 1

res.drop(['t'], axis = 1, inplace=True)
res.to_csv('all_integration_unloadings.csv', na_rep='NA', index=False)