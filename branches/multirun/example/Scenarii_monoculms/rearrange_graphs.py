import os
import shutil
import pandas as pd

## Graphs to be rearranged
graphs_titles_inputs = ['N_content_roots_axis','N_content_shoot_axis','N_content_blade','shoot_roots_ratio_axis','green_area_blade','Ag_blade',
                 'lamina_Wmax','leaf_Lmax','internode_Lmax','sheath_Lmax','phyllochron','Conc_Sucrose_phloem','Conc_Amino_acids_phloem', 'S_Amino_Acids_roots','LAI','RER_comparison',
                 'Conc_Nitrates_roots','Conc_Sucrose_roots','sum_dry_mass_roots_axis','sum_dry_mass_shoot_axis','sum_dry_mass_axis','Summary','SSLW','RUE','SLN_blade','SLN_nonstruct_blade',
                 'C_usages_cumulated','Photosynthetic_yield_blade','Photosynthetic_yield_plante','Photosynthetic_yield_plante2','Ts_blade','internode_Lmax', 'NNI_axis','N_dilution',
                        'S_Amino_Acids_blade','HATS_LATS_roots']


## Get the list of scenarii
titi = os.walk('.').next()[1]
scenarii = []
for i in titi:
    if i[:9] == 'Scenario_':
        scenarii.append( int( i[9:] ) )
scenarii_inputs = scenarii

## Get the scenario labels
# scenarii_df = pd.read_csv( os.path.join('inputs','scenarii_list.csv'), index_col='Scenario')
# scenarii_df['Scenario'] = scenarii_df.index
# if 'Scenario_label' not in scenarii_df.keys():
#     scenarii_df['Scenario_label'] =  ''
# else:
#     scenarii_df['Scenario_label'] = scenarii_df['Scenario_label'].fillna('')

def rearrange_graphs(graphs_titles=None, scenarii=None):
    """
    For each graph type, create a common directory with the graphs of all the scenarii.

    :param list graphs_titles: List of graph title to be rearranged.
    :param list scenarii: List of scenarii (numbers) for which the graphs will be rearranged.
    """

    if scenarii is None:
        scenarii = scenarii_inputs
    if graphs_titles is None:
        graphs_titles = graphs_titles_inputs

    # Create directory with graphs by type
    graph_dir = os.path.join('Graphs_by_type')
    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)

    #: For each graph type, create a common directory
    for graph in graphs_titles:

        graph_dir = os.path.join('Graphs_by_type',graph)
        if not os.path.exists(graph_dir):
            os.mkdir(graph_dir)

        #: For each scenario, copy and rename the graphs in the common directory
        for scenario in scenarii:

            scenario_name = 'Scenario_' + str(scenario)
            # scenario_label = scenarii_df['Scenario_label'].get( scenario, '' ).replace(" ", "_")

            scenario_dir = os.path.join(scenario_name, 'graphs')
            graph_src = os.path.join(scenario_dir, graph+'.PNG')
            graph_dest = os.path.join(graph_dir, graph+'.PNG')
            graph_renamed = os.path.join(graph_dir, scenario_name+'.PNG') #os.path.join(graph_dir, scenario_name+'_'+scenario_label+'.PNG')

            if os.path.exists(graph_renamed):
                os.remove(graph_renamed)
            if os.path.exists(graph_src):
                shutil.copyfile(graph_src, graph_dest)
                os.rename(graph_dest,graph_renamed)

if __name__ == '__main__':
    rearrange_graphs()