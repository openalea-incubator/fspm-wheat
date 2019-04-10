# -*- coding: latin-1 -*-

from tempwheat import converter, simulation
import tools


"""
    fspmwheat.tempwheat_facade
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    The module :mod:`fspmwheat.tempwheat_facade` is a facade of the model TempWheat.

    This module permits to initialize and run the model TempWheat from a :class:`MTG <openalea.mtg.mtg.MTG>`
    in a convenient and transparent way, wrapping all the internal complexity of the model, and dealing
    with all the tedious initialization and conversion processes.

    :copyright: Copyright 2014-2016 INRA-ECOSYS, see AUTHORS.
    :license: TODO, see LICENSE for details.

    .. seealso:: Barillot et al. 2016.
"""

"""
    Information about this versioned file:
        $LastChangedBy: rbarillot $
        $LastChangedDate: 2018-08-10 18:07:45 +0200 (ven., 10 août 2018) $
        $LastChangedRevision: 38 $
        $URL: https://subversion.renater.fr/fspm-wheat/trunk/fspmwheat/tempwheat_facade.py $
        $Id: tempwheat_facade.py 38 2018-08-10 16:07:45Z rbarillot $
"""

SHARED_SAM_INPUTS_OUTPUTS_INDEXES = ['plant', 'axis']

class TempWheatFacade(object):
    """
    The TempWheatFacade class permits to initialize, run the model TempWheat
    from a :class:`MTG <openalea.mtg.mtg.MTG>`, and update the MTG and the dataframes
    shared between all models.

    Use :meth:`run` to run the model.

    :Parameters:

        - `shared_mtg` (:class:`openalea.mtg.mtg.MTG`) - The MTG shared between all models.
        - `delta_t` (:class:`int`) - The delta between two runs, in seconds.
        - `model_SAM_inputs_df` (:class:`pandas.DataFrame`) - the inputs of the model at SAM (axis) scale.
        - `shared_SAM_inputs_outputs_df` (:class:`pandas.DataFrame`) - the dataframe of inputs and outputs at SAM (axis) scale shared between all models.

           This model must implement a method `add_metamer(mtg, plant_index, axis_label)` to add a metamer to a specific axis of a plant in a MTG.
    """

    def __init__(self, shared_mtg, delta_t,
                 model_SAM_inputs_df,
                 shared_SAM_inputs_outputs_df):

        self._shared_mtg = shared_mtg  #: the MTG shared between all models

        self._simulation = simulation.Simulation(delta_t=delta_t)  #: the simulator to use to run the model

        all_tempwheat_inputs_dict = converter.from_dataframes(model_SAM_inputs_df)
        self._update_shared_MTG( all_tempwheat_inputs_dict['SAM'])

        self._shared_SAM_inputs_outputs_df = shared_SAM_inputs_outputs_df  #: the dataframe at SAM scale shared between all models
        self._update_shared_dataframes(model_SAM_inputs_df)

    def run(self, Tair, Tsol):
        """
        Run the model and update the MTG and the dataframes shared between all models.

        :Parameters:

            - `Tair` (:class:`float`) - air temperature at t (degree Celsius)
            - `Tsol` (:class:`float`) - soil temperature at t (degree Celsius)
        """
        self._initialize_model()
        self._simulation.run(Tair, Tsol)
        self._update_shared_MTG(self._simulation.outputs['SAM'])
        tempwheat_SAM_outputs_df = converter.to_dataframes(self._simulation.outputs)

        self._update_shared_dataframes(tempwheat_SAM_outputs_df)

    def _initialize_model(self):
        """
        Initialize the inputs of the model from the MTG shared between all models.
        """
        all_tempwheat_SAM_dict = {}

        for mtg_plant_vid in self._shared_mtg.components_iter(self._shared_mtg.root):
            mtg_plant_index = int(self._shared_mtg.index(mtg_plant_vid))

            # Axis scale
            for mtg_axis_vid in self._shared_mtg.components_iter(mtg_plant_vid):
                mtg_axis_label = self._shared_mtg.label(mtg_axis_vid)
                mtg_axis_properties = self._shared_mtg.get_vertex_property(mtg_axis_vid)

                # SAM
                if 'SAM' in mtg_axis_properties:
                    SAM_vid = (mtg_plant_index, mtg_axis_label)
                    mtg_SAM_properties = mtg_axis_properties['SAM']
                    tempwheat_SAM_inputs_dict = {}

                    is_valid_SAM = True
                    for SAM_input_name in simulation.SAM_INPUTS:
                        if SAM_input_name in mtg_SAM_properties:
                            # use the input from the MTG
                            tempwheat_SAM_inputs_dict[SAM_input_name] = mtg_SAM_properties[SAM_input_name]
                        else:
                            is_valid_SAM = False
                            break
                    if is_valid_SAM:
                        all_tempwheat_SAM_dict[SAM_vid] = tempwheat_SAM_inputs_dict

        self._simulation.initialize({'SAM': all_tempwheat_SAM_dict})

    def _update_shared_MTG(self, all_tempwheat_SAM_data_dict):
        """
        Update the MTG shared between all models from the inputs or the outputs of the model.
        """
        # add the properties if needed
        mtg_property_names = self._shared_mtg.property_names()
        for tempwheat_data_name in set(simulation.SAM_INPUTS_OUTPUTS):
            if tempwheat_data_name not in mtg_property_names:
                self._shared_mtg.add_property(tempwheat_data_name)

        if 'SAM' not in mtg_property_names:
            self._shared_mtg.add_property('SAM')

        # update the properties of the MTG
        for mtg_plant_vid in self._shared_mtg.components_iter(self._shared_mtg.root):
            mtg_plant_index = int(self._shared_mtg.index(mtg_plant_vid))

            # Axis scale
            for mtg_axis_vid in self._shared_mtg.components_iter(mtg_plant_vid):
                mtg_axis_label = self._shared_mtg.label(mtg_axis_vid)
                SAM_id = (mtg_plant_index, mtg_axis_label)
                if SAM_id in all_tempwheat_SAM_data_dict:
                    tempwheat_SAM_data_dict = all_tempwheat_SAM_data_dict[SAM_id]
                    mtg_axis_properties = self._shared_mtg.get_vertex_property(mtg_axis_vid)
                    if 'SAM' not in mtg_axis_properties:
                        self._shared_mtg.property('SAM')[mtg_axis_vid] = {}
                    for SAM_data_name, SAM_data_value in tempwheat_SAM_data_dict.items():
                        self._shared_mtg.property('SAM')[mtg_axis_vid][SAM_data_name] = SAM_data_value


    def _update_shared_dataframes(self, tempwheat_SAM_data_df):
        """
        Update the dataframes shared between all models from the inputs dataframes or the outputs dataframes of the model.
        """

        tempwheat_data_df, shared_inputs_outputs_indexes, shared_inputs_outputs_df = tempwheat_SAM_data_df, SHARED_SAM_INPUTS_OUTPUTS_INDEXES, self._shared_SAM_inputs_outputs_df

        tools.combine_dataframes_inplace(tempwheat_data_df, shared_inputs_outputs_indexes, shared_inputs_outputs_df)
