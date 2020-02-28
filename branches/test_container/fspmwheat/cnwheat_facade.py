# -*- coding: latin-1 -*-

from respiwheat import model as respiwheat_model

from cnwheat import model as cnwheat_model, simulation as cnwheat_simulation, \
    converter as cnwheat_converter, postprocessing as cnwheat_postprocessing, parameters as cnwheat_parameters

from fspmwheat import tools

import numpy as np
import math

"""
    fspmwheat.cnwheat_facade
    ~~~~~~~~~~~~~~~~~~~~~~~~

    The module :mod:`fspmwheat.cnwheat_facade` is a facade of the model CNWheat.

    This module permits to initialize and run the model CNWheat from a :class:`MTG <openalea.mtg.mtg.MTG>`
    in a convenient and transparent way, wrapping all the internal complexity of the model, and dealing
    with all the tedious initialization and conversion processes.

    :copyright: Copyright 2014-2016 INRA-ECOSYS, see AUTHORS.
    :license: see LICENSE for details.

"""

"""
    Information about this versioned file:
        $LastChangedBy$
        $LastChangedDate$
        $LastChangedRevision$
        $URL$
        $Id$
"""

#: the mapping of CNWheat organ classes to the attributes in axis and phytomer which represent an organ
CNWHEAT_ATTRIBUTES_MAPPING = {cnwheat_model.Internode: 'internode', cnwheat_model.Lamina: 'lamina',
                              cnwheat_model.Sheath: 'sheath', cnwheat_model.Peduncle: 'peduncle', cnwheat_model.Chaff: 'chaff',
                              cnwheat_model.Roots: 'roots', cnwheat_model.Grains: 'grains', cnwheat_model.Phloem: 'phloem',
                              cnwheat_model.HiddenZone: 'hiddenzone'}

#: the mapping of organs (which belong to an axis) labels in MTG to organ classes in CNWheat
MTG_TO_CNWHEAT_AXES_ORGANS_MAPPING = {'grains': cnwheat_model.Grains, 'phloem': cnwheat_model.Phloem, 'roots': cnwheat_model.Roots}

#: the mapping of organs (which belong to a phytomer) labels in MTG to organ classes in CNWheat
MTG_TO_CNWHEAT_PHYTOMERS_ORGANS_MAPPING = {'internode': cnwheat_model.Internode, 'blade': cnwheat_model.Lamina, 'sheath': cnwheat_model.Sheath, 'peduncle': cnwheat_model.Peduncle,
                                           'ear': cnwheat_model.Chaff, 'hiddenzone': cnwheat_model.HiddenZone}

# # the mapping of CNWheat photosynthetic organs to CNWheat photosynthetic organ elements
CNWHEAT_ORGANS_TO_ELEMENTS_MAPPING = {cnwheat_model.Internode: cnwheat_model.InternodeElement, cnwheat_model.Lamina: cnwheat_model.LaminaElement, cnwheat_model.Sheath: cnwheat_model.SheathElement,
                                      cnwheat_model.Peduncle: cnwheat_model.PeduncleElement, cnwheat_model.Chaff: cnwheat_model.ChaffElement}

#: the parameters and variables which define the state of a CNWheat population
POPULATION_STATE_VARIABLE = set(cnwheat_simulation.Simulation.PLANTS_STATE + cnwheat_simulation.Simulation.AXES_STATE +
                                cnwheat_simulation.Simulation.PHYTOMERS_STATE + cnwheat_simulation.Simulation.ORGANS_STATE +
                                cnwheat_simulation.Simulation.HIDDENZONE_STATE + cnwheat_simulation.Simulation.ELEMENTS_STATE)

# number of seconds in 1 hour
HOUR_TO_SECOND_CONVERSION_FACTOR = 3600


class CNWheatFacade(object):
    """
    The CNWheatFacade class permits to initialize, run the model CNWheat
    from a :class:`MTG <openalea.mtg.mtg.MTG>`, and update the MTG and the dataframes
    shared between all models.

    Use :meth:`run` to run the model.

    """

    def __init__(self, shared_mtg, delta_t, culm_density, update_parameters,
                 model_organs_inputs_df,
                 model_hiddenzones_inputs_df,
                 model_elements_inputs_df,
                 model_soils_inputs_df,
                 shared_axes_inputs_outputs_df,
                 shared_organs_inputs_outputs_df,
                 shared_hiddenzones_inputs_outputs_df,
                 shared_elements_inputs_outputs_df,
                 shared_soils_inputs_outputs_df):
        """
        :param openalea.mtg.mtg.MTG shared_mtg: The MTG shared between all models.
        :param int delta_t: The delta between two runs, in seconds.
        :param dict culm_density: The density of culm. One key per plant.
        :param dict update_parameters: A dictionary with the parameters to update, should have the form {'Organ_label1': {'param1': value1, 'param2': value2}, ...}.
        :param pandas.DataFrame model_organs_inputs_df: the inputs of the model at organs scale.
        :param pandas.DataFrame model_hiddenzones_inputs_df: the inputs of the model at hiddenzones scale.
        :param pandas.DataFrame model_elements_inputs_df: the inputs of the model at elements scale.
        :param pandas.DataFrame model_soils_inputs_df: the inputs of the model at soils scale.
        :param pandas.DataFrame shared_axes_inputs_outputs_df: the dataframe of inputs and outputs at axes scale shared between all models.
        :param pandas.DataFrame shared_organs_inputs_outputs_df: the dataframe of inputs and outputs at organs scale shared between all models.
        :param pandas.DataFrame shared_hiddenzones_inputs_outputs_df: the dataframe of inputs and outputs at hiddenzones scale shared between all models.
        :param pandas.DataFrame shared_elements_inputs_outputs_df: the dataframe of inputs and outputs at elements scale shared between all models.
        :param pandas.DataFrame shared_soils_inputs_outputs_df: the dataframe of inputs and outputs at soils scale shared between all models.

        """

        self._shared_mtg = shared_mtg  #: the MTG shared between all models

        self._simulation = cnwheat_simulation.Simulation(respiration_model=respiwheat_model, delta_t=delta_t, culm_density=culm_density)

        self.population, self.soils = cnwheat_converter.from_dataframes(model_organs_inputs_df, model_hiddenzones_inputs_df, model_elements_inputs_df, model_soils_inputs_df)

        self._update_parameters = update_parameters

        self._simulation.initialize(self.population, self.soils)

        self._update_shared_MTG()

        self._shared_axes_inputs_outputs_df = shared_axes_inputs_outputs_df  #: the dataframe at axes scale shared between all models
        self._shared_organs_inputs_outputs_df = shared_organs_inputs_outputs_df  #: the dataframe at organs scale shared between all models
        self._shared_hiddenzones_inputs_outputs_df = shared_hiddenzones_inputs_outputs_df  #: the dataframe at hiddenzones scale shared between all models
        self._shared_elements_inputs_outputs_df = shared_elements_inputs_outputs_df  #: the dataframe at elements scale shared between all models
        self._shared_soils_inputs_outputs_df = shared_soils_inputs_outputs_df  #: the dataframe at soils scale shared between all models
        self._update_shared_dataframes(cnwheat_organs_data_df=model_organs_inputs_df,
                                       cnwheat_hiddenzones_data_df=model_hiddenzones_inputs_df,
                                       cnwheat_elements_data_df=model_elements_inputs_df,
                                       cnwheat_soils_data_df=model_soils_inputs_df)

    def run(self, Tair=12, Tsoil=12, tillers_replications=None):
        """
        Run the model and update the MTG and the dataframes shared between all models.

        :param float Tair: air temperature (�C)
        :param float Tsoil: soil temperature (�C)
        :param dict [str, float] tillers_replications: a dictionary with tiller id as key, and weight of replication as value.
        """

        self._initialize_model(Tair=Tair, Tsoil=Tsoil, tillers_replications=tillers_replications)
        self._simulation.run()
        self._update_shared_MTG()

        _, cnwheat_axes_inputs_outputs_df, _, cnwheat_organs_inputs_outputs_df, cnwheat_hiddenzones_inputs_outputs_df, cnwheat_elements_inputs_outputs_df, cnwheat_soils_inputs_outputs_df = \
            cnwheat_converter.to_dataframes(self._simulation.population, self._simulation.soils)

        self._update_shared_dataframes(cnwheat_axes_data_df=cnwheat_axes_inputs_outputs_df,
                                       cnwheat_organs_data_df=cnwheat_organs_inputs_outputs_df,
                                       cnwheat_hiddenzones_data_df=cnwheat_hiddenzones_inputs_outputs_df,
                                       cnwheat_elements_data_df=cnwheat_elements_inputs_outputs_df,
                                       cnwheat_soils_data_df=cnwheat_soils_inputs_outputs_df)

    @staticmethod
    def postprocessing(axes_outputs_df, organs_outputs_df, hiddenzone_outputs_df, elements_outputs_df, soils_outputs_df, delta_t):
        """
        Run the postprocessing.

        :param pandas.DataFrame axes_outputs_df: the outputs of the model at axis scale.
        :param pandas.DataFrame organs_outputs_df: the outputs of the model at organ scale.
        :param pandas.DataFrame hiddenzone_outputs_df: the outputs of the model at hiddenzone scale.
        :param pandas.DataFrame elements_outputs_df: the outputs of the model at element scale.
        :param pandas.DataFrame soils_outputs_df: the outputs of the model at element scale.
        :param int delta_t: The delta between two runs, in seconds.

    :return: post-processing for each scale:
            * plant (see :attr:`PLANTS_RUN_POSTPROCESSING_VARIABLES`)
            * axis (see :attr:`AXES_RUN_POSTPROCESSING_VARIABLES`)
            * metamer (see :attr:`PHYTOMERS_RUN_POSTPROCESSING_VARIABLES`)
            * organ (see :attr:`ORGANS_RUN_POSTPROCESSING_VARIABLES`)
            * hidden zone (see :attr:`HIDDENZONE_RUN_POSTPROCESSING_VARIABLES`)
            * element (see :attr:`ELEMENTS_RUN_POSTPROCESSING_VARIABLES`)
            * and soil (see :attr:`SOILS_RUN_POSTPROCESSING_VARIABLES`)
        depending of the dataframes given as argument.
        For example, if user passes only dataframes `plants_df`, `axes_df` and `metamers_df`,
        then only post-processing dataframes of plants, axes and metamers are returned.
    :rtype: tuple [pandas.DataFrame]
        """

        (_, _, organs_postprocessing_df,
         elements_postprocessing_df,
         hiddenzones_postprocessing_df,
         axes_postprocessing_df,
         soils_postprocessing_df) = cnwheat_postprocessing.postprocessing(axes_df=axes_outputs_df, hiddenzones_df=hiddenzone_outputs_df,
                                                                          organs_df=organs_outputs_df, elements_df=elements_outputs_df,
                                                                          soils_df=soils_outputs_df, delta_t=delta_t)
        return axes_postprocessing_df, hiddenzones_postprocessing_df, organs_postprocessing_df, elements_postprocessing_df, soils_postprocessing_df

    @staticmethod
    def graphs(axes_postprocessing_df, hiddenzones_postprocessing_df, organs_postprocessing_df, elements_postprocessing_df, soils_postprocessing_df, graphs_dirpath='.'):
        """
        Generate the graphs and save them into `graphs_dirpath`.

        :param pandas.DataFrame axes_postprocessing_df: CN-Wheat outputs at axis scale
        :param pandas.DataFrame hiddenzones_postprocessing_df: CN-Wheat outputs at hidden zone scale
        :param pandas.DataFrame organs_postprocessing_df: CN-Wheat outputs at organ scale
        :param pandas.DataFrame elements_postprocessing_df: CN-Wheat outputs at element scale
        :param pandas.DataFrame soils_postprocessing_df: CN-Wheat outputs at soil scale
        :param str graphs_dirpath: the path of the directory to save the generated graphs in

        """
        cnwheat_postprocessing.generate_graphs(axes_df=axes_postprocessing_df,
                                               hiddenzones_df=hiddenzones_postprocessing_df,
                                               organs_df=organs_postprocessing_df,
                                               elements_df=elements_postprocessing_df,
                                               soils_df=soils_postprocessing_df,
                                               graphs_dirpath=graphs_dirpath)

    def _initialize_model(self, Tair=12, Tsoil=12, tillers_replications=None):
        """
        Initialize the inputs of the model from the MTG shared between all models and the soils.

        :param float Tair: air temperature (�C)
        :param float Tsoil: soil temperature (�C)
        :param dict [str, float] tillers_replications: a dictionary with tiller id as key, and weight of replication as value.
        """

        # Convert number of replications per tiller into number of replications per cohort
        cohorts_replications = {}
        if tillers_replications is not None:
            for tiller_id, replication_weight in tillers_replications.items():
                try:
                    tiller_rank = int(tiller_id[1:])
                except:
                    continue
                cohorts_replications[tiller_rank + 3] = replication_weight

        self.population = cnwheat_model.Population()

        # traverse the MTG recursively from top
        for mtg_plant_vid in self._shared_mtg.components_iter(self._shared_mtg.root):
            mtg_plant_index = int(self._shared_mtg.index(mtg_plant_vid))
            # create a new plant
            cnwheat_plant = cnwheat_model.Plant(mtg_plant_index)
            is_valid_plant = False

            for mtg_axis_vid in self._shared_mtg.components_iter(mtg_plant_vid):
                mtg_axis_label = self._shared_mtg.label(mtg_axis_vid)

                #: Hack to treat tillering cases : TEMPORARY
                if mtg_axis_label != 'MS':
                    try:
                        tiller_rank = int(mtg_axis_label[1:])
                    except:
                        continue
                    cnwheat_plant.cohorts.append(tiller_rank + 3)

                #: MS
                # create a new axis
                cnwheat_axis = cnwheat_model.Axis(mtg_axis_label)
                is_valid_axis = True
                for cnwheat_organ_class in (cnwheat_model.Roots, cnwheat_model.Phloem, cnwheat_model.Grains):
                    mtg_organ_label = cnwheat_converter.CNWHEAT_CLASSES_TO_DATAFRAME_ORGANS_MAPPING[cnwheat_organ_class]
                    # create a new organ
                    cnwheat_organ = cnwheat_organ_class(mtg_organ_label)
                    mtg_axis_properties = self._shared_mtg.get_vertex_property(mtg_axis_vid)
                    if mtg_organ_label in mtg_axis_properties:
                        mtg_organ_properties = mtg_axis_properties[mtg_organ_label]
                        cnwheat_organ_data_names = set(cnwheat_simulation.Simulation.ORGANS_STATE).intersection(cnwheat_organ.__dict__)
                        if set(mtg_organ_properties).issuperset(cnwheat_organ_data_names):
                            cnwheat_organ_data_dict = {}
                            for cnwheat_organ_data_name in cnwheat_organ_data_names:
                                cnwheat_organ_data_dict[cnwheat_organ_data_name] = mtg_organ_properties[cnwheat_organ_data_name]

                                # MG
                                if math.isnan(mtg_organ_properties[cnwheat_organ_data_name]) or mtg_organ_properties[cnwheat_organ_data_name] is None:
                                    print(mtg_axis_vid)
                                    print(mtg_organ_label)
                                    print(cnwheat_organ_data_name)

                            cnwheat_organ.__dict__.update(cnwheat_organ_data_dict)

                            # Update parameters if specified
                            if mtg_organ_label in self._update_parameters:
                                cnwheat_organ.PARAMETERS.__dict__.update(self._update_parameters[mtg_organ_label])

                            cnwheat_organ.initialize()
                            # add the new organ to current axis
                            setattr(cnwheat_axis, mtg_organ_label, cnwheat_organ)
                        elif cnwheat_organ_class is not cnwheat_model.Grains:
                            is_valid_axis = False
                            break
                    elif cnwheat_organ_class is not cnwheat_model.Grains:
                        is_valid_axis = False
                        break

                if not is_valid_axis:
                    continue

                has_valid_phytomer = False
                for mtg_metamer_vid in self._shared_mtg.components_iter(mtg_axis_vid):
                    mtg_metamer_index = int(self._shared_mtg.index(mtg_metamer_vid))

                    # create a new phytomer
                    cnwheat_phytomer = cnwheat_model.Phytomer(mtg_metamer_index, cohorts=cnwheat_plant.cohorts, cohorts_replications=cohorts_replications)  #: Hack to treat tillering cases :TEMPORARY

                    mtg_hiddenzone_label = cnwheat_converter.CNWHEAT_CLASSES_TO_DATAFRAME_ORGANS_MAPPING[cnwheat_model.HiddenZone]
                    mtg_metamer_properties = self._shared_mtg.get_vertex_property(mtg_metamer_vid)

                    if mtg_hiddenzone_label in mtg_metamer_properties:
                        mtg_hiddenzone_properties = mtg_metamer_properties[mtg_hiddenzone_label]
                        if set(mtg_hiddenzone_properties).issuperset(cnwheat_simulation.Simulation.HIDDENZONE_STATE):
                            has_valid_hiddenzone = True
                            cnwheat_hiddenzone_data_dict = {}
                            for cnwheat_hiddenzone_data_name in cnwheat_simulation.Simulation.HIDDENZONE_STATE:
                                cnwheat_hiddenzone_data_dict[cnwheat_hiddenzone_data_name] = mtg_hiddenzone_properties[cnwheat_hiddenzone_data_name]

                            # create a new hiddenzone
                            cnwheat_hiddenzone = cnwheat_model.HiddenZone(mtg_hiddenzone_label, cohorts=cnwheat_plant.cohorts, cohorts_replications=cohorts_replications, index=cnwheat_phytomer.index,
                                                                          **cnwheat_hiddenzone_data_dict)

                            # Update parameters if specified
                            if mtg_hiddenzone_label in self._update_parameters:
                                cnwheat_hiddenzone.PARAMETERS.__dict__.update(self._update_parameters[mtg_hiddenzone_label])

                            cnwheat_hiddenzone.initialize()
                            # add the new hiddenzone to current phytomer
                            setattr(cnwheat_phytomer, mtg_hiddenzone_label, cnwheat_hiddenzone)
                        else:
                            has_valid_hiddenzone = False
                    else:
                        has_valid_hiddenzone = False

                    has_valid_organ = False
                    for mtg_organ_vid in self._shared_mtg.components_iter(mtg_metamer_vid):
                        mtg_organ_label = self._shared_mtg.label(mtg_organ_vid)
                        if mtg_organ_label not in MTG_TO_CNWHEAT_PHYTOMERS_ORGANS_MAPPING or self._shared_mtg.get_vertex_property(mtg_organ_vid)['length'] == 0:
                            continue

                        # create a new organ
                        cnwheat_organ_class = MTG_TO_CNWHEAT_PHYTOMERS_ORGANS_MAPPING[mtg_organ_label]
                        cnwheat_organ = cnwheat_organ_class(mtg_organ_label)

                        # Update parameters if specified
                        if 'PhotosyntheticOrgan' in self._update_parameters:
                            cnwheat_organ.PARAMETERS.__dict__.update(self._update_parameters['PhotosyntheticOrgan'])

                        cnwheat_organ.initialize()
                        has_valid_element = False

                        # Create a new element
                        for mtg_element_vid in self._shared_mtg.components_iter(mtg_organ_vid):
                            mtg_element_properties = self._shared_mtg.get_vertex_property(mtg_element_vid)
                            mtg_element_label = self._shared_mtg.label(mtg_element_vid)
                            if mtg_element_label not in cnwheat_converter.DATAFRAME_TO_CNWHEAT_ELEMENTS_NAMES_MAPPING \
                                    or (self._shared_mtg.get_vertex_property(mtg_element_vid)['length'] == 0) \
                                    or (self._shared_mtg.get_vertex_property(mtg_element_vid).get('mstruct', 0) == 0) \
                                    or ((mtg_element_label == 'HiddenElement') and (self._shared_mtg.get_vertex_property(mtg_element_vid).get('is_growing', True))) \
                                    or (self._shared_mtg.get_vertex_property(mtg_element_vid).get('green_area', 0) <= 0.25E-6):
                                continue  # TODO: Check that we are not taking out some relevant cases with the condition on mstruct == 0

                            has_valid_element = True
                            cnwheat_element_data_dict = {}
                            for cnwheat_element_data_name in cnwheat_simulation.Simulation.ELEMENTS_STATE:
                                mtg_element_data_value = mtg_element_properties.get(cnwheat_element_data_name)
                                # In case the value is None, or the proprety is not even defined, we take default value from InitCompartment
                                if mtg_element_data_value is None or np.isnan(mtg_element_data_value):
                                    if cnwheat_element_data_name == 'Ts':
                                        mtg_element_data_value = Tair
                                    else:
                                        mtg_element_data_value = cnwheat_parameters.PhotosyntheticOrganElementInitCompartments().__dict__[cnwheat_element_data_name]
                                cnwheat_element_data_dict[cnwheat_element_data_name] = mtg_element_data_value
                            cnwheat_element = CNWHEAT_ORGANS_TO_ELEMENTS_MAPPING[cnwheat_organ_class](mtg_element_label, cohorts=cnwheat_plant.cohorts, cohorts_replications=cohorts_replications,
                                                                                                      index=cnwheat_phytomer.index, **cnwheat_element_data_dict)
                            # Add parameters from organ scale
                            cnwheat_element.PARAMETERS.__dict__.update(cnwheat_organ.PARAMETERS.__dict__)

                            # add the new element to current organ
                            setattr(cnwheat_organ, cnwheat_converter.DATAFRAME_TO_CNWHEAT_ELEMENTS_NAMES_MAPPING[mtg_element_label], cnwheat_element)

                        if has_valid_element:
                            has_valid_organ = True
                            setattr(cnwheat_phytomer, CNWHEAT_ATTRIBUTES_MAPPING[cnwheat_organ_class], cnwheat_organ)

                    if has_valid_organ or has_valid_hiddenzone:
                        cnwheat_axis.phytomers.append(cnwheat_phytomer)
                        has_valid_phytomer = True

                if not has_valid_phytomer:
                    is_valid_axis = False

                if is_valid_axis:
                    cnwheat_plant.axes.append(cnwheat_axis)
                    is_valid_plant = True

            if is_valid_plant:
                self.population.plants.append(cnwheat_plant)

        self._simulation.initialize(self.population, self.soils, Tair=Tair, Tsoil=Tsoil)

    def _update_shared_MTG(self):
        """
        Update the MTG shared between all models from the population of CNWheat.
        """
        # add the missing properties
        mtg_property_names = self._shared_mtg.property_names()
        for cnwheat_data_name in POPULATION_STATE_VARIABLE:
            if cnwheat_data_name not in mtg_property_names:
                self._shared_mtg.add_property(cnwheat_data_name)
        for cnwheat_organ_label in MTG_TO_CNWHEAT_AXES_ORGANS_MAPPING.keys() + [cnwheat_converter.CNWHEAT_CLASSES_TO_DATAFRAME_ORGANS_MAPPING[cnwheat_model.HiddenZone]]:
            if cnwheat_organ_label not in mtg_property_names:
                self._shared_mtg.add_property(cnwheat_organ_label)

        mtg_plants_iterator = self._shared_mtg.components_iter(self._shared_mtg.root)
        # traverse CN-Wheat population from top
        for cnwheat_plant in self.population.plants:
            cnwheat_plant_index = cnwheat_plant.index
            while True:
                mtg_plant_vid = next(mtg_plants_iterator)
                if int(self._shared_mtg.index(mtg_plant_vid)) == cnwheat_plant_index:
                    break
            mtg_axes_iterator = self._shared_mtg.components_iter(mtg_plant_vid)
            for cnwheat_axis in cnwheat_plant.axes:
                cnwheat_axis_label = cnwheat_axis.label
                while True:
                    mtg_axis_vid = next(mtg_axes_iterator)
                    if self._shared_mtg.label(mtg_axis_vid) == cnwheat_axis_label:
                        break
                for mtg_organ_label in MTG_TO_CNWHEAT_AXES_ORGANS_MAPPING.keys():
                    if mtg_organ_label not in self._shared_mtg.get_vertex_property(mtg_axis_vid):
                        # Add a property describing the organ to the current axis of the MTG
                        self._shared_mtg.property(mtg_organ_label)[mtg_axis_vid] = {}
                    # Update the property describing the organ of the current axis in the MTG
                    cnwheat_organ = getattr(cnwheat_axis, mtg_organ_label)
                    mtg_organ_properties = self._shared_mtg.get_vertex_property(mtg_axis_vid)[mtg_organ_label]
                    for cnwheat_property_name in cnwheat_simulation.Simulation.ORGANS_STATE:
                        if hasattr(cnwheat_organ, cnwheat_property_name):
                            mtg_organ_properties[cnwheat_property_name] = getattr(cnwheat_organ, cnwheat_property_name)
                mtg_metamers_iterator = self._shared_mtg.components_iter(mtg_axis_vid)
                for cnwheat_phytomer in cnwheat_axis.phytomers:
                    cnwheat_phytomer_index = cnwheat_phytomer.index
                    while True:
                        mtg_metamer_vid = next(mtg_metamers_iterator)
                        if int(self._shared_mtg.index(mtg_metamer_vid)) == cnwheat_phytomer_index:
                            break
                    if cnwheat_phytomer.hiddenzone is not None:
                        mtg_hiddenzone_label = cnwheat_converter.CNWHEAT_CLASSES_TO_DATAFRAME_ORGANS_MAPPING[cnwheat_model.HiddenZone]
                        if mtg_hiddenzone_label not in self._shared_mtg.get_vertex_property(mtg_metamer_vid):
                            # Add a property describing the hiddenzone to the current metamer of the MTG
                            self._shared_mtg.property(mtg_hiddenzone_label)[mtg_metamer_vid] = {}
                        # Update the property describing the hiddenzone of the current metamer in the MTG
                        mtg_hiddenzone_properties = self._shared_mtg.get_vertex_property(mtg_metamer_vid)[mtg_hiddenzone_label]
                        mtg_hiddenzone_properties.update(cnwheat_phytomer.hiddenzone.__dict__)
                    for mtg_organ_vid in self._shared_mtg.components_iter(mtg_metamer_vid):
                        mtg_organ_label = self._shared_mtg.label(mtg_organ_vid)
                        if mtg_organ_label not in MTG_TO_CNWHEAT_PHYTOMERS_ORGANS_MAPPING: continue
                        cnwheat_organ = getattr(cnwheat_phytomer, CNWHEAT_ATTRIBUTES_MAPPING[MTG_TO_CNWHEAT_PHYTOMERS_ORGANS_MAPPING[mtg_organ_label]])
                        if cnwheat_organ is None: continue
                        cnwheat_organ_property_names = [property_name for property_name in cnwheat_simulation.Simulation.ORGANS_STATE if hasattr(cnwheat_organ, property_name)]
                        for cnwheat_organ_property_name in cnwheat_organ_property_names:
                            attribute_value = getattr(cnwheat_organ, cnwheat_organ_property_name)
                            # TODO: temporary ; replace by inputs at photosynthetic organs scale
                            if attribute_value is not None:
                                self._shared_mtg.property(cnwheat_organ_property_name)[mtg_organ_vid] = attribute_value
                            elif cnwheat_organ_property_name not in self._shared_mtg.get_vertex_property(mtg_organ_vid):
                                self._shared_mtg.property(cnwheat_organ_property_name)[mtg_organ_vid] = attribute_value

                        for mtg_element_vid in self._shared_mtg.components_iter(mtg_organ_vid):
                            mtg_element_label = self._shared_mtg.label(mtg_element_vid)
                            if mtg_element_label not in cnwheat_converter.DATAFRAME_TO_CNWHEAT_ELEMENTS_NAMES_MAPPING: continue
                            cnwheat_element = getattr(cnwheat_organ, cnwheat_converter.DATAFRAME_TO_CNWHEAT_ELEMENTS_NAMES_MAPPING[mtg_element_label])
                            cnwheat_element_property_names = [property_name for property_name in cnwheat_simulation.Simulation.ELEMENTS_STATE if hasattr(cnwheat_element, property_name)]
                            for cnwheat_element_property_name in cnwheat_element_property_names:
                                cnwheat_element_property_value = getattr(cnwheat_element, cnwheat_element_property_name)
                                self._shared_mtg.property(cnwheat_element_property_name)[mtg_element_vid] = cnwheat_element_property_value
                                self._shared_mtg.property(cnwheat_element_property_name)[mtg_organ_vid] = cnwheat_element_property_value  # Update organ property too

    def _update_shared_dataframes(self, cnwheat_axes_data_df=None, cnwheat_organs_data_df=None,
                                  cnwheat_hiddenzones_data_df=None, cnwheat_elements_data_df=None,
                                  cnwheat_soils_data_df=None):
        """
        Update the dataframes shared between all models from the inputs dataframes or the outputs dataframes of the cnwheat model.

        :param pandas.DataFrame cnwheat_axes_data_df: CN-Wheat shared dataframe at axis scale
        :param pandas.DataFrame cnwheat_organs_data_df: CN-Wheat shared dataframe at organ scale
        :param pandas.DataFrame cnwheat_hiddenzones_data_df: CN-Wheat shared dataframe hiddenzone scale
        :param pandas.DataFrame cnwheat_elements_data_df: CN-Wheat shared dataframe at element scale
        :param pandas.DataFrame cnwheat_soils_data_df: CN-Wheat shared dataframe at soil scale
        """

        for cnwheat_data_df, \
            shared_inputs_outputs_indexes, \
            shared_inputs_outputs_df in ((cnwheat_axes_data_df, cnwheat_simulation.Simulation.AXES_INDEXES, self._shared_axes_inputs_outputs_df),
                                         (cnwheat_organs_data_df, cnwheat_simulation.Simulation.ORGANS_INDEXES, self._shared_organs_inputs_outputs_df),
                                         (cnwheat_hiddenzones_data_df, cnwheat_simulation.Simulation.HIDDENZONE_INDEXES, self._shared_hiddenzones_inputs_outputs_df),
                                         (cnwheat_elements_data_df, cnwheat_simulation.Simulation.ELEMENTS_INDEXES, self._shared_elements_inputs_outputs_df),
                                         (cnwheat_soils_data_df, cnwheat_simulation.Simulation.SOILS_INDEXES, self._shared_soils_inputs_outputs_df)):

            if cnwheat_data_df is None: continue

            tools.combine_dataframes_inplace(cnwheat_data_df, shared_inputs_outputs_indexes, shared_inputs_outputs_df)
