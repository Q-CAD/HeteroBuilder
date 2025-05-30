from vdW_structures.vdW_structure import VdWStructure
from vdW_structures.unique_structures import UniqueStructureGetter
from typing import Union, List
from scipy.linalg import sqrtm
import numpy as np
import math
import os
import json
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder


class VdWHeterostructureGenerator():
    def __init__(self, **kwargs):
        self.sm = StructureMatcher(**kwargs)
        pass

    def polar_decomposition(self, film_sl_vectors, substrate_sl_vectors):
        s1, s2 = substrate_sl_vectors[0][:2], substrate_sl_vectors[1][:2]
        f1, f2 = film_sl_vectors[0][:2], film_sl_vectors[1][:2]

        S = np.column_stack([s1, s2])   # shape (2,2)
        F = np.column_stack([f1, f2])   # shape (2,2)

        M = F @ np.linalg.inv(S)
        U = sqrtm(M.T @ M)    # shape (2,2), symmetric by construction
        R = M @ np.linalg.inv(U)
        angle_rad = np.arctan2(R[1,0], R[0,0])
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def compute_signed_film_substrate_angle(self, film_sl_vectors, substrate_sl_vectors):
        """
        Compute the signed angle between the film and substrate motifs, distinguishing rotations.

        Args:
            film_sl_vectors (list of list): Superlattice vectors of the film (2D list of 3D vectors).
            substrate_sl_vectors (list of list): Superlattice vectors of the substrate (2D list of 3D vectors).

        Returns:
            float: Signed angle in degrees between the film and substrate repeating motifs.
        """
        # Convert to numpy arrays and take the first in-plane vectors
        film_v1 = np.array(film_sl_vectors[0][:2])  # Only consider x, y components
        sub_v1 = np.array(substrate_sl_vectors[0][:2])

        # Compute angle using atan2 for proper signed angle determination
        dot_product = np.dot(film_v1, sub_v1)
        cross_product = np.cross(np.append(film_v1, 0), np.append(sub_v1, 0))  # Append z=0 for cross product
        angle_radians = np.arctan2(cross_product[2], dot_product)  # Use z-component of cross product

        # Convert to degrees
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees  # Returns values from -180 to 180 degrees
    
    def sort_by_structure_length(self, vdW_heterostructures: List[VdWStructure], heterostructure_properties: List):
        '''Sorts structures by the number of atoms present'''
        zipped_heterostructures = zip(vdW_heterostructures, heterostructure_properties)
        sorted_zipped_heterostructures = sorted(zipped_heterostructures, key=lambda x: len(x[0].structure))
        s_vdW_heterostructures, s_heterostructure_properties = zip(*sorted_zipped_heterostructures)

        return s_vdW_heterostructures, s_heterostructure_properties

    def get_unique_structures(self, structures: List[Structure], **kwargs):
        '''Function that returns only the unique structures (based on PMG StructureMatcher) in a list of pymatgen Structure objects'''
        sm = StructureMatcher(**kwargs)
        unique_structures = []
        for structure in tqdm(structures):
            is_match = False
            for unique_structure in unique_structures:
                matched = sm.fit(structure, unique_structure)
                if matched == True:
                    is_match = True
                else:
                    continue
            if is_match == False:
                unique_structures.append(structure)
        return unique_structures
    
    def generate_vdW_heterostructures(self, 
                                      film: VdWStructure, 
                                      substrate: VdWStructure,
                                      zsl_generator: ZSLGenerator,
                                      get_unique: bool=True,
                                      n_chunks: int=4,   
                                      **kwargs):
        '''Uses Zur's algorithm as implemented in pymatgen's ZSLGenerator to create heterostructure lattices'''

        # 
        if 'vacuum_over_film' not in kwargs:
            vacuum_over_film = substrate.vdW_spacings[0]
        else:
            vacuum_over_film = kwargs['vacuum_over_film']
            kwargs.pop('vacuum_over_film', None)

        if 'interface_spacing' not in kwargs:
            interface_spacing = film.vdW_spacings[0]
        else:
            interface_spacing = kwargs['interface_spacing']
            kwargs.pop('interface_spacing', None)
        
        cib = CoherentInterfaceBuilder(substrate_structure=substrate.structure, 
                                       film_structure=film.structure, 
                                       zslgen=zsl_generator, 
                                       film_miller=(0, 0, 1), 
                                       substrate_miller=(0, 0, 1))

        # Do not attempt to find new terminations; set shifts to 0
        default_key, default_value = ('default', 'default'), (0, 0)
        cib.terminations = [default_key]
        cib._terminations[default_key] = default_value
        interfaces = cib.get_interfaces(default_key, gap=interface_spacing, vacuum_over_film=vacuum_over_film) # Generator object

        # Get the unique_interfaces and return them as pymatgen Structure objects
        interfaces_list = list(interfaces)
        if get_unique:
            usg = UniqueStructureGetter(**kwargs)
            interfaces_list = usg.filter_unique_with_recursive_chunks(interfaces_list, n_chunks=n_chunks)
        
        # Get the smallest know gap to use as the minimum vdW spacing
        threshold_tolerance = 1e-4
        minimum_vdW_gap = np.min(film.vdW_spacings + substrate.vdW_spacings + [interface_spacing]) - threshold_tolerance
        unique_vdW_heterostructures = [VdWStructure(
            interface, minimum_vdW_gap) for interface in interfaces_list]
        heterostructure_properties = [interface.interface_properties for interface in interfaces_list]

        # Compute the signed and decomposition angles between interfaces and add to heterostructure_properties dictionary
        for heterostructure_property in heterostructure_properties:
            heterostructure_property['signed_angle'] = self.compute_signed_film_substrate_angle(heterostructure_property['film_sl_vectors'], 
                                                                                                heterostructure_property['substrate_sl_vectors'])  
            heterostructure_property['decomposition_angle'] = self.polar_decomposition(heterostructure_property['film_sl_vectors'],
                                                                                                heterostructure_property['substrate_sl_vectors'])

        return self.sort_by_structure_length(unique_vdW_heterostructures, heterostructure_properties)

    def write_heterostructures_data(self, directory, vdW_heterostructures, heterostructure_properties):
        
        def convert_ndarrays_to_lists(d):
            """
            Recursively convert all numpy ndarray objects in a dictionary to lists.

            Args:
                d (dict): The input dictionary.

            Returns:
                dict: The dictionary with all np.ndarray objects converted to lists.
            """
            if isinstance(d, dict):
                return {key: convert_ndarrays_to_lists(value) for key, value in d.items()}
            elif isinstance(d, list):
                return [convert_ndarrays_to_lists(item) for item in d]
            elif isinstance(d, np.ndarray):
                return d.tolist()  # Convert numpy array to list
            else:
                return d  # Return other types unchanged

        len_dct = {}
        for i, vdW_heterostructure in enumerate(vdW_heterostructures):
            length = len(vdW_heterostructure.structure)
            if length in len_dct.keys():
                len_dct[length] += 1
            else:
                len_dct[length] = 0
            
            write_path = os.path.join(directory, str(length), str(len_dct[length]))
            os.makedirs(write_path, exist_ok=True)
            vdW_heterostructure.structure.to(filename=os.path.join(write_path, 'POSCAR'))
            
            with open(os.path.join(write_path, 'interface.json'), 'w') as f:
                json.dump(convert_ndarrays_to_lists(heterostructure_properties[i]), 
                          f, indent=4)       

        return 
