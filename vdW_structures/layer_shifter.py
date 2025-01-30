from ase import Atoms
import numpy as np
from ase.geometry.geometry import get_layers
import numpy



class LayerShifter:
    def __init__(self):
        """Allows layer (all atoms layers above atom layer) and atom layer shifts in the z-direction."""
        pass

    def reciprocal_lattice_vectors(self, a, b, c):
        V = np.dot(a, np.cross(b, c))
        a_star = 2 * np.pi * np.cross(b, c) / V
        b_star = 2 * np.pi * np.cross(c, a) / V
        c_star = 2 * np.pi * np.cross(a, b) / V
        return a_star, b_star, c_star

    def miller_indices_to_normal(self, h, k, l, a, b, c):
        a_star, b_star, c_star = self.reciprocal_lattice_vectors(a, b, c)
        n = h * a_star + k * b_star + l * c_star
        n_unit = n / np.linalg.norm(n)
        return n_unit

    def layer_distance(self, layer_levels, layer_number):
        return layer_levels[layer_number] - layer_levels[layer_number-1]

    def get_shift_vector(self, atoms, miller_indice, current_distance, desired_distance):
        normal_vector = self.miller_indices_to_normal(miller_indice[0], miller_indice[1], 
                                             miller_indice[2], atoms.cell[0],
                                             atoms.cell[1], atoms.cell[2])
        shift_vector = (current_distance - desired_distance) * normal_vector
        return shift_vector

    def shift_atoms(self, atoms, miller_indice, layer_number, layer_indices, current_distance, desired_distance):
        shift_vector = self.get_shift_vector(atoms, miller_indice, current_distance, desired_distance)
        new_positions = []

        for i, atom in enumerate(atoms):
            if layer_indices[i] < layer_number:
                new_positions.append(np.add(atoms[i].position, shift_vector))
            else:
                new_positions.append(atoms[i].position)
        shifted_atoms = Atoms(positions=new_positions, symbols=atoms.symbols, cell=atoms.cell, pbc=True)
        return shifted_atoms

    def shift_layer(self, atoms, miller_indice, layer_number, layer_indices, shift_distance):
        normal_vector= self.miller_indices_to_normal(miller_indice[0], miller_indice[1],
                                             miller_indice[2], atoms.cell[0],
                                             atoms.cell[1], atoms.cell[2])
        shift_vector = shift_distance * normal_vector
        new_positions = []

        for i, atom in enumerate(atoms):
            if layer_indices[i] == layer_number:
                new_positions.append(np.add(atoms[i].position, shift_vector))
            else:
                new_positions.append(atoms[i].position)
        shifted_atoms = Atoms(positions=new_positions, symbols=atoms.symbols, cell=atoms.cell, pbc=True)
        return shifted_atoms

    def print_layers(self, atoms, miller_list):
        layer_indices, layer_levels = get_layers(atoms, tuple(miller_list))
        num_layers = np.max(layer_indices) + 1
        print(f'Structure has {num_layers} layers')
        for layer_num in range(0, num_layers):
            indices = [i for i, val in enumerate(layer_indices) if val == layer_num]
            elements = [atoms[i].symbol for i in indices]
            if layer_num == 0:
                distance = 0
            else:
                distance = np.round(self.layer_distance(layer_levels, layer_num), 5) 
            print(f'Layer {layer_num}: {elements}{indices}, spacing: {distance}')
        return

    def structure_shift(self, atoms: Atoms, 
                        layer_number=0, 
                        distance=0, 
                        shift_layers=False, 
                        shift_one=False, 
                        verbose=False):
        
        if not isinstance(atoms, Atoms):
            raise TypeError(f"Expected an ase Atoms object, got {type(atoms).__name__}")
            
        shifted_atoms = None
        miller_list = [0, 0, 1]
        layer_indices, layer_levels = get_layers(atoms, tuple(miller_list))
        num_layers = np.max(layer_indices) + 1
        if shift_layers is True:
            current_distance = self.layer_distance(layer_levels, layer_number)
            shifted_atoms = self.shift_atoms(atoms, miller_list, layer_number, layer_indices, current_distance, distance)
            if verbose:
                self.print_layers(shifted_atoms, miller_list)
        elif shift_one is True:
            shifted_atoms = self.shift_layer(atoms, miller_list, layer_number, layer_indices, distance)
            if verbose:
                self.print_layers(shifted_atoms, miller_list)
        else:
            if verbose:
                self.print_layers(atoms, miller_list)
        return shifted_atoms