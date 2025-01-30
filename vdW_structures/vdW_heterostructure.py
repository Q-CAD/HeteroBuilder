from vdW_structure import VdWStructure
from multiprocessing import Pool
from functools import partial
from typing import Union, List
import numpy as np
import math
from tqdm import tqdm
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder



class VdWHeterostructureGenerator():
    def __init__(self, **kwargs):
        self.sm = StructureMatcher(**kwargs)
        pass

    def is_similar_to_any(self, structure, unique_structures):
        for unique_structure in unique_structures:
            if self.sm.fit(structure, unique_structure):
                return True
        return False

    def filter_unique_structures(self, structures: List[Structure]):
        unique_structures = []
        for structure in structures:
            with Pool() as pool:
                similar = pool.starmap(self.is_similar_to_any, 
                                       [(structure, unique_structures)] * len(structures))
            if not any(similar):
                unique_structures.append(structure)
        return unique_structures

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
    
    def generate_unique_vdW_heterostructures(self, 
                                      film: VdWStructure, 
                                      substrate: VdWStructure,
                                      zsl_generator: ZSLGenerator,  
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
        unique_interfaces = self.get_unique_structures(interfaces, **kwargs)

        # Get the smallest know gap to use as the minimum vdW spacing
        threshold_tolerance = 1e-4
        minimum_vdW_gap = np.min(film.vdW_spacings + substrate.vdW_spacings + [interface_spacing]) - threshold_tolerance
        unique_vdW_heterostructures = [VdWStructure(
            unique_interface, minimum_vdW_gap) for unique_interface in unique_interfaces]
        return unique_vdW_heterostructures