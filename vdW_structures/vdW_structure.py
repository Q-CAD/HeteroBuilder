from typing import Union, List
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.lattice import Lattice
from ase.geometry.geometry import get_layers
from vdW_structures.layer_shifter import LayerShifter
from pymatgen.io.ase import AseAtomsAdaptor
from copy import deepcopy


class VdWStructure:
    def __init__(self, structure: Structure, minimum_vdW_gap: Union[int, float]):
        """Identify van der Waals layers based on if the spacing between atomic layers exceeds the minimum_vdW_gap."""
        if not isinstance(structure, Structure):
            raise TypeError(f"Expected a pymatgen Structure object, got {type(structure).__name__}")
        if not isinstance(minimum_vdW_gap, (float, int)):
            raise TypeError(f"Expected a float/int for minimum_vdW_gap, got {type(minimum_vdW_gap).__name__}")

        self.minimum_vdW_gap = minimum_vdW_gap
        
        self.vdW_layers, self.layer_images, self.vdW_spacings = self.get_vdW_layers(structure)
        #print(self.vdW_layers, self.layer_images, self.vdW_spacings)
        
        if len(self.vdW_layers) == 1 and self.vdW_spacings[0] < self.minimum_vdW_gap:
            raise ValueError(f"No van der Waals layers found with a gap of {minimum_vdW_gap} Å.")
        
        self.structure = self.shift_to_vdW_gap(structure, self.vdW_layers, self.layer_images).get_sorted_structure()
        self.vdW_layers, self.layer_images, self.vdW_spacings = self.get_vdW_layers(self.structure)
        #print(self.vdW_layers, self.layer_images, self.vdW_spacings)
        
        self.ase_atoms_adaptor = AseAtomsAdaptor()
        self.layer_shifter = LayerShifter()

    def shift_site(self, site, shift, pos_ind, coords_are_cartesian):
        shift_vector = [0, 0, 0]
        shift_vector[pos_ind] = shift
        if coords_are_cartesian:
            coordinates = np.add(site.coords, shift_vector)
        else:
            coordinates = np.add(site.frac_coords, shift_vector)
        return coordinates
    
    def shift_vdW_layers(self, layer_inds: Union[int, List[int]], 
                        shift: Union[Union[int, float], List[Union[int, float]]], 
                        pos_ind: int, 
                        coords_are_cartesian: bool=True):
        """Shifts vdW_layers provided by layer_inds by a shift amount"""

        if type(layer_inds) is not list:
            layer_inds = [layer_inds]
        if type(shift) is not list:
            shift = [shift]

        if len(layer_inds) != len(shift):
            raise ValueError(f"layer_inds and shift arguments are different lengths!")

        shifts_dct = {}
        for ii, i in enumerate(layer_inds):
            for j in self.vdW_layers[i]:
                shifts_dct[j] = shift[ii]

        new_coordinates = []
        for site_ind in range(len(self.structure)):
            if site_ind in list(shifts_dct.keys()):
                new_coordinates.append(self.shift_site(self.structure[site_ind], shifts_dct[site_ind], 
                                                       pos_ind, coords_are_cartesian))
            else:
                new_coordinates.append(self.shift_site(self.structure[site_ind], 0, 
                                                       pos_ind, coords_are_cartesian)) # no shift
        shifted_structure = Structure(
            lattice=self.structure.lattice,
            coords=new_coordinates,
            species=[site.specie for site in self.structure],
            coords_are_cartesian=coords_are_cartesian, 
            to_unit_cell=True
        )
        
        if not shifted_structure.is_valid():
            print(f"Warning: the applied shift(s) has made this structure invalid. Proceed with caution!")
        
        return VdWStructure(shifted_structure.get_sorted_structure(), self.minimum_vdW_gap)
    
    def x_shift_vdW_layers(self, layer_inds: Union[int, List[int]], shift: Union[int, float], coords_are_cartesian: bool=True):
        return self.shift_vdW_layers(layer_inds, shift, 0, coords_are_cartesian) 

    def y_shift_vdW_layers(self, layer_inds: Union[int, List[int]], shift: Union[int, float], coords_are_cartesian: bool=True):
        return self.shift_vdW_layers(layer_inds, shift, 1, coords_are_cartesian)

    def z_shift_vdW_layers(self, layer_inds: Union[int, List[int]], shift: Union[int, float], coords_are_cartesian: bool=True):
        return self.shift_vdW_layers(layer_inds, shift, 2, coords_are_cartesian)

    def z_solve_shift_vdW_layers(self, layer_inds: Union[int, List[int]], solve_shift: Union[int, float], adjust_z_cell: bool=True):
        ''' Solves for a CARTESIAN shift towards the vdW layer below the structure that yields the solve_shift interlayer distance '''
        structure_copy = deepcopy(self.structure)
        atoms_copy = structure_copy.to_ase_atoms()
        
        atom_layer_indices, atom_layer_levels = get_layers(atoms_copy, (0, 0, 1))
        
        if type(layer_inds) is not list:
            layer_inds = [layer_inds]
        if type(solve_shift) is not list:
            solve_shift = [solve_shift]

        all_shift_vectors = [0, 0, 0]
        bottom_shift = None
        for ii, i in enumerate(layer_inds):
            vdW_atom_layers = [atom_layer_indices[j] for j in self.vdW_layers[i]]
            min_vdW_atom_layer_number = np.min(vdW_atom_layers) # Get the minimum layer
            if min_vdW_atom_layer_number == 0: # Handle bottom layer shift differently
                bottom_shift = solve_shift[ii]              
            else:
                atoms_copy = self.layer_shifter.structure_shift(atoms_copy, 
                                                            layer_number=min_vdW_atom_layer_number, 
                                                            distance=solve_shift[ii],
                                                            shift_layers=True, shift_one=False
                )
                current_distance = self.layer_shifter.layer_distance(atom_layer_levels, min_vdW_atom_layer_number)
                shift_vector = self.layer_shifter.get_shift_vector(atoms_copy, (0, 0, 1), current_distance, solve_shift[ii])
                all_shift_vectors = np.subtract(all_shift_vectors, shift_vector)

        # Increases the z_cell vector magnitude by summed total of the shifts, along the z_cell vector direction
        shifted_structure = self.ase_atoms_adaptor.get_structure(atoms_copy)
        if adjust_z_cell == True:
            shifted_structure_lattice = Lattice(np.array([
                                                shifted_structure.lattice.matrix[0],
                                                shifted_structure.lattice.matrix[1],
                                                np.add(shifted_structure.lattice.matrix[2], np.array(all_shift_vectors))
            ]))

            shifted_structure = Structure(lattice=shifted_structure_lattice,
                                         coords=[site.coords for site in shifted_structure],
                                         species=[site.specie for site in shifted_structure],
                                         coords_are_cartesian=True,
                                         to_unit_cell=False
            )

        if bottom_shift:
            shifted_atoms = shifted_structure.to_ase_atoms()
            shifted_atoms.center(vacuum=0.5 * bottom_shift, axis=2)
            shifted_structure = self.ase_atoms_adaptor.get_structure(shifted_atoms)

        if not shifted_structure.is_valid():
            print(f"Warning: the applied shift(s) has made this structure invalid. Proceed with caution!")
            
        return VdWStructure(shifted_structure.get_sorted_structure(), self.minimum_vdW_gap)

    def set_vacuum(self, vacuum: Union[int, float]):
        structure_copy = deepcopy(self.structure)
        atoms_copy = self.structure.to_ase_atoms()
        atoms_copy.center(vacuum=0.5 * vacuum, axis=2)
        vacuum_structure = self.ase_atoms_adaptor.get_structure(atoms_copy)
        
        return VdWStructure(vacuum_structure.get_sorted_structure(), self.minimum_vdW_gap)

    def get_vdW_layers(self, structure: Structure):
        """Identify van der Waals layers in the structure based on spacing along the z-direction."""
        vdW_layers = [[]]
        z_images = [[]]
        vdW_spacings = []

        z_magnitude = structure.lattice.c
        all_zs, all_indices = [s.coords[2] for s in structure], [i for i, site in enumerate(structure)]
        zipped = zip(all_zs, all_indices)
        s_zipped = sorted(zipped, key=lambda x: x[0])
        zs, idxs = zip(*s_zipped)
        diffs = [z_magnitude - zs[-1] + zs[0]] + [zs[i+1] - zs[i] for i in range(len(zs)-1)]

        count = 0
        for i, diff in enumerate(diffs):
            if diff > self.minimum_vdW_gap:
                vdW_spacings.append(diff)
                vdW_layers.append([])
                z_images.append([])
                count += 1
            vdW_layers[count].append(idxs[i])
            z_images[count].append(0)

        # Handle periodic boundary conditions
        if count > 0 and diffs[0] < self.minimum_vdW_gap:
            z_images[0] = [-1] * len(vdW_layers[0]) + [0] * len(vdW_layers[-1]) # Change image number here
            vdW_layers[0] += vdW_layers[-1]
            vdW_layers.pop()
            z_images.pop()

        # Clean up empty layers
        cleaned_layers = [layer for layer in vdW_layers if layer]
        cleaned_images = [z_images[i] for i, layer in enumerate(vdW_layers) if layer]
    
        return cleaned_layers, cleaned_images, vdW_spacings

    def shift_to_vdW_gap(self, structure: Structure, layers: List[int], images: List[int]):
        """Shift the structure to ensure periodic boundaries fall in vdW gaps."""
        for layer_index, layer in enumerate(layers):
            for i, site_ind in enumerate(layer):
                if images[layer_index][i] != 0:
                    shift_factor = np.negative(images[layer_index][i])
                    shift = np.multiply(shift_factor, structure.lattice.matrix[2])
                    structure[site_ind].coords += shift

        zs = [site.coords[2] for site in structure]
        padding = (structure.lattice.c - (max(zs) - min(zs))) / 2
        z_shift = structure.lattice.c - padding - max(zs)

        for site in structure:
            site.coords += z_shift

        return Structure(
            lattice=structure.lattice,
            coords=[site.coords for site in structure],
            species=[site.specie for site in structure],
            coords_are_cartesian=True,
            to_unit_cell=True
        ).get_sorted_structure()

    def _add_vdW_sites(self, new_layer_numbers: List[int]):
        """Adds z-direction spacing to the new lattice and adds sites with appropriate spacing."""
        new_sites = []
        new_lattice = Lattice(np.array([
            self.structure.lattice.matrix[0],
            self.structure.lattice.matrix[1],
            np.multiply(len(new_layer_numbers) + 1, self.structure.lattice.matrix[2])
        ]))

        for new_layer_number_ind, new_layer_number in enumerate(new_layer_numbers):
            for site_number in self.vdW_layers[new_layer_number]:
                site_shift = np.multiply(new_layer_number_ind, self.structure.lattice.matrix[2])
                coords = self.structure[site_number].coords + site_shift
                species = self.structure[site_number].species
                site = PeriodicSite(
                    species=species, coords=coords, lattice=new_lattice, coords_are_cartesian=True
                )
                new_sites.append(site)

        new_structure = Structure(
            lattice=new_lattice,
            coords=[site.coords for site in new_sites],
            species=[site.specie for site in new_sites],
            coords_are_cartesian=True,
            to_unit_cell=True
        )
        return new_structure.get_sorted_structure()

    def _rescale_vdW_lattice(self, new_layer_numbers: List[int], new_structure: Structure):
        """Correct the vdW spacing in the new structure with the extended lattice."""
        new_structure_atoms = new_structure.to_ase_atoms()
        new_layers, new_shifts, new_spacings = self.get_vdW_layers(new_structure)
        new_indices, new_levels = get_layers(new_structure_atoms, (0, 0, 1))

        # Perform the correct layer shifts for each layer
        for upper_layer_ind, upper_layer_number in enumerate(new_layers[1:]):  # Skip bottom layer
            shift_layer_number = new_indices[upper_layer_number[0]]
            layer_to_get_shift = new_layer_numbers[upper_layer_ind + 1]
            shift = self.vdW_spacings[layer_to_get_shift]
            
            new_structure_atoms = self.layer_shifter.structure_shift(
                new_structure_atoms, layer_number=shift_layer_number, distance=shift,
                shift_layers=True, shift_one=False
            )
        
        # Create the appropriate Lattice and Structure
        new_structure = self.ase_atoms_adaptor.get_structure(new_structure_atoms)
        newest_indices, newest_levels = get_layers(new_structure_atoms, (0, 0, 1))
        z_magnitude = newest_levels[-1] - newest_levels[0] + self.vdW_spacings[new_layer_numbers[0]]
        newest_z = z_magnitude * (self.structure.lattice.matrix[2] / np.linalg.norm(self.structure.lattice.matrix[2]))
        newest_lattice = Lattice(np.array([
            self.structure.lattice.matrix[0],
            self.structure.lattice.matrix[1],
            newest_z
        ]))

        coord_shift = [0, 0, 0.5 * self.vdW_spacings[new_layer_numbers[0]] - new_structure[0].coords[2]]
        newest_structure = Structure(
            lattice=newest_lattice,
            coords=[site.coords + coord_shift for site in new_structure],
            species=[site.specie for site in new_structure],
            coords_are_cartesian=True,
            to_unit_cell=True
        )
        
        return newest_structure.get_sorted_structure()

    def extend_vdW_layers(self, new_layer_numbers: List[int]):
        """Extend the vdW layers by duplicating specified layers into a new structure. Returns VdWBulk."""
        for new_layer_number in new_layer_numbers:
            if new_layer_number not in range(len(self.vdW_layers)):
                raise IndexError(f"vdW layer index {new_layer_number} is not valid for vdW_layers of len({len(self.vdW_layers)})!")
                
        no_spacing_extended_structure = self._add_vdW_sites(new_layer_numbers)
        correct_spacing_structure = self._rescale_vdW_lattice(new_layer_numbers, no_spacing_extended_structure)
        return VdWStructure(correct_spacing_structure.get_sorted_structure(), self.minimum_vdW_gap)
