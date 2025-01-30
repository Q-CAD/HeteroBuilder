from vdW_structures.vdW_structure import VdWStructure
from vdW_structures.unique_structures import UniqueStructureGetter
from typing import Union, List
import numpy as np
import math
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder



class VdWHeterostructureGenerator():
    def __init__(self, **kwargs):
        self.sm = StructureMatcher(**kwargs)
        pass

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
        
        return unique_vdW_heterostructures
