# -*- coding: latin-1 -*-

import warnings

from farquharwheat import converter, simulation

import tools

from alinea.astk.plantgl_utils import get_height  # for height calculation

import numpy as np


"""
    fspmwheat.farquharwheat_facade
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    The module :mod:`fspmwheat.farquharwheat_facade` is a facade of the model FarquharWheat.

    This module permits to initialize and run the model FarquharWheat from a :class:`MTG <openalea.mtg.mtg.MTG>`
    in a convenient and transparent way, wrapping all the internal complexity of the model, and dealing
    with all the tedious initialization and conversion processes.

    :copyright: Copyright 2014-2016 INRA-ECOSYS, see AUTHORS.
    :license: TODO, see LICENSE for details.

    .. seealso:: Barillot et al. 2016.
"""

"""
    Information about this versioned file:
        $LastChangedBy$
        $LastChangedDate$
        $LastChangedRevision$
        $URL$
        $Id$
"""


#: the name of the organs modeled by FarquharWheat
FARQUHARWHEAT_ORGANS_NAMES = {'internode', 'blade', 'sheath', 'peduncle', 'ear'}

#: names of the elements
FARQUHARWHEAT_ELEMENTS_INPUTS = ['HiddenElement', 'StemElement', 'LeafElement1']
FARQUHARWHEAT_VISIBLE_ELEMENTS_INPUTS = ['StemElement', 'LeafElement1']

#: the columns which define the topology in the elements scale dataframe shared between all models
SHARED_ELEMENTS_INPUTS_OUTPUTS_INDEXES = ['plant', 'axis', 'metamer', 'organ', 'element']


class FarquharWheatFacade(object):
    """
    The FarquharWheatFacade class permits to initialize, run the model FarquharWheat
    from a :class:`MTG <openalea.mtg.mtg.MTG>`, and update the MTG and the dataframes
    shared between all models.

    Use :meth:`run` to run the model.

    :Parameters:

        - `shared_mtg` (:class:`openalea.mtg.mtg.MTG`) - The MTG shared between all models.
        - `model_elements_inputs_df` (:class:`pandas.DataFrame`) - the inputs of the model at elements scale.
        - `shared_elements_inputs_outputs_df` (:class:`pandas.DataFrame`) - the dataframe of inputs and outputs at elements scale shared between all models.
    """

    def __init__(self, shared_mtg,
                 model_elements_inputs_df,
                 model_SAMs_inputs_df,
                 shared_elements_inputs_outputs_df):

        self._shared_mtg = shared_mtg  #: the MTG shared between all models

        self._simulation = simulation.Simulation()  #: the simulator to use to run the model

        all_farquharwheat_inputs_dict = converter.from_dataframe(model_elements_inputs_df, model_SAMs_inputs_df)
        self._update_shared_MTG(all_farquharwheat_inputs_dict)

        self._shared_elements_inputs_outputs_df = shared_elements_inputs_outputs_df  #: the dataframe at elements scale shared between all models
        self._update_shared_dataframes(model_elements_inputs_df)

    def run(self, Ta, ambient_CO2, RH, Ur):
        """
        Run the model and update the MTG and the dataframes shared between all models.

        :Parameters:

            - `Ta` (:class:`float`) - air temperature at t (degree Celsius)

            - `ambient_CO2` (:class:`float`) - air CO2 at t (�mol mol-1)

            - `RH` (:class:`float`) - relative humidity at t (decimal fraction)

            - `Ur` (:class:`float`) - wind speed at the top of the canopy at t (m s-1)

        """
        self._initialize_model()
        self._simulation.run(Ta, ambient_CO2, RH, Ur)
        self._update_shared_MTG(self._simulation.outputs)
        farquharwheat_elements_outputs_df = converter.to_dataframe(self._simulation.outputs)
        self._update_shared_dataframes(farquharwheat_elements_outputs_df)

    def _initialize_model(self):
        """
        Initialize the inputs of the model from the MTG shared between all models.
        """
        all_farquharwheat_elements_inputs_dict = {}
        all_farquharwheat_SAM_inputs_dict = {}

        # traverse the MTG recursively from top ...
        for mtg_plant_vid in self._shared_mtg.components_iter(self._shared_mtg.root):
            mtg_plant_index = int(self._shared_mtg.index(mtg_plant_vid))
            for mtg_axis_vid in self._shared_mtg.components_iter(mtg_plant_vid):
                mtg_axis_label = self._shared_mtg.label(mtg_axis_vid)
                SAM_id = (mtg_plant_index, mtg_axis_label)
                farquharwheat_SAM_inputs_dict = {}
                for farquharwheat_SAM_input_name in converter.FARQUHARWHEAT_SAMS_INPUTS:
                    farquharwheat_SAM_inputs_dict[farquharwheat_SAM_input_name] = self._shared_mtg.get_vertex_property(mtg_axis_vid)['SAM'].get(farquharwheat_SAM_input_name)

                height_element_list = []

                for mtg_metamer_vid in self._shared_mtg.components_iter(mtg_axis_vid):
                    mtg_metamer_index = int(self._shared_mtg.index(mtg_metamer_vid))
                    for mtg_organ_vid in self._shared_mtg.components_iter(mtg_metamer_vid):
                        mtg_organ_label = self._shared_mtg.label(mtg_organ_vid)
                        mtg_organ_length = self._shared_mtg.get_vertex_property(mtg_organ_vid).get('length', 0)
                        if mtg_organ_label not in FARQUHARWHEAT_ORGANS_NAMES or mtg_organ_length <= 0: continue

                        for mtg_element_vid in self._shared_mtg.components_iter(mtg_organ_vid):
                            mtg_element_properties = self._shared_mtg.get_vertex_property(mtg_element_vid)
                            mtg_element_label = self._shared_mtg.label(mtg_element_vid)
                            mtg_element_length = self._shared_mtg.get_vertex_property(mtg_element_vid).get('length', 0)

                            if mtg_element_label not in FARQUHARWHEAT_ELEMENTS_INPUTS or mtg_element_length <= 0: continue  # to excluse topElement, baseElement and elements with null length
                            element_id = (mtg_plant_index, mtg_axis_label, mtg_metamer_index, mtg_organ_label, mtg_element_label)

                            farquharwheat_element_inputs_dict = {}
                            # TODO: temporary ; replace 'FARQUHARWHEAT_ELEMENT_PROPERTIES_TEMP' by default values in a parameters file
                            FARQUHARWHEAT_ELEMENT_PROPERTIES_TEMP = {'PARa': 0, 'nitrates': 0, 'amino_acids': 0, 'proteins': 0, 'Nstruct': 0, 'green_area': 0}

                            for farquharwheat_element_input_name in converter.FARQUHARWHEAT_ELEMENTS_INPUTS:
                                mtg_element_input = mtg_element_properties.get(farquharwheat_element_input_name)
                                if mtg_element_input is None:
                                    mtg_element_input = FARQUHARWHEAT_ELEMENT_PROPERTIES_TEMP.get(farquharwheat_element_input_name)
                                #: Height computation for growing visible elements
                                if mtg_element_label in FARQUHARWHEAT_VISIBLE_ELEMENTS_INPUTS and farquharwheat_element_input_name == 'height':
                                    mtg_element_geom = self._shared_mtg.property('geometry').get(mtg_element_vid)
                                    if mtg_element_geom is not None:  # It seems like visible elements with very little area don't have geometry. # TODO : Ckeck ADEL's area threshold for geometry representation
                                        triangle_heights = get_height({mtg_element_vid: self._shared_mtg.property('geometry')[mtg_element_vid]})
                                        mtg_element_input = np.nanmean(triangle_heights[mtg_element_vid])
                                    else:
                                        mtg_element_input = None
                                    height_element_list.append(mtg_element_input)
                                #: Width is actually diameter for Sheath and Internodes
                                if mtg_element_label == 'StemElement' and farquharwheat_element_input_name == 'width':
                                    mtg_element_input = mtg_element_properties.get('diameter', 0.0)

                                farquharwheat_element_inputs_dict[farquharwheat_element_input_name] = mtg_element_input

                            all_farquharwheat_elements_inputs_dict[element_id] = farquharwheat_element_inputs_dict

                farquharwheat_SAM_inputs_dict['height_canopy'] = np.nanmax(np.array(height_element_list, dtype=np.float64))
                if np.isnan(farquharwheat_SAM_inputs_dict['height_canopy']):
                    farquharwheat_SAM_inputs_dict['height_canopy'] = 0.78  # TODO : by default values in a parameters file
                all_farquharwheat_SAM_inputs_dict[SAM_id] = farquharwheat_SAM_inputs_dict

        self._simulation.initialize({'elements': all_farquharwheat_elements_inputs_dict, 'SAMs': all_farquharwheat_SAM_inputs_dict})

    def _update_shared_MTG(self, farquharwheat_data_dict):
        """
        Update the MTG shared between all models from the inputs or the outputs of the model.
        """
        # add the properties if needed
        mtg_property_names = self._shared_mtg.property_names()
        for farquharwheat_elements_data_name in converter.FARQUHARWHEAT_ELEMENTS_INPUTS_OUTPUTS:
            if farquharwheat_elements_data_name not in mtg_property_names:
                self._shared_mtg.add_property(farquharwheat_elements_data_name)

        # traverse the MTG recursively from top ...
        for mtg_plant_vid in self._shared_mtg.components_iter(self._shared_mtg.root):
            mtg_plant_index = int(self._shared_mtg.index(mtg_plant_vid))
            for mtg_axis_vid in self._shared_mtg.components_iter(mtg_plant_vid):
                mtg_axis_label = self._shared_mtg.label(mtg_axis_vid)
                for mtg_metamer_vid in self._shared_mtg.components_iter(mtg_axis_vid):
                    mtg_metamer_index = int(self._shared_mtg.index(mtg_metamer_vid))
                    for mtg_organ_vid in self._shared_mtg.components_iter(mtg_metamer_vid):
                        mtg_organ_label = self._shared_mtg.label(mtg_organ_vid)
                        if mtg_organ_label not in FARQUHARWHEAT_ORGANS_NAMES: continue
                        for mtg_element_vid in self._shared_mtg.components_iter(mtg_organ_vid):
                            mtg_element_label = self._shared_mtg.label(mtg_element_vid)
                            element_id = (mtg_plant_index, mtg_axis_label, mtg_metamer_index, mtg_organ_label, mtg_element_label)
                            if element_id not in farquharwheat_data_dict: continue
                            # update the element in the MTG
                            farquharwheat_element_data_dict = farquharwheat_data_dict[element_id]
                            for farquharwheat_element_data_name, farquharwheat_element_data_value in farquharwheat_element_data_dict.items():
                                self._shared_mtg.property(farquharwheat_element_data_name)[mtg_element_vid] = farquharwheat_element_data_value

    def _update_shared_dataframes(self, farquharwheat_elements_data_df):
        """
        Update the dataframes shared between all models from the inputs dataframes or the outputs dataframes of the model.
        """
        tools.combine_dataframes_inplace(farquharwheat_elements_data_df, SHARED_ELEMENTS_INPUTS_OUTPUTS_INDEXES, self._shared_elements_inputs_outputs_df)
