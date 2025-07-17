import py3Dmol
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from pymatgen.core.periodic_table import Element 

def view_structure(structure, color_map, look='b'):
    view = py3Dmol.view(width=800, height=800)
    cif_str = structure.to(fmt="cif")
    view.addModel(cif_str, "cif")

    # Apply sphere style and color for each element
    for element, color in color_map.items():
        view.setStyle({"elem": element}, {"sphere": {"radius": Element(element).atomic_radius, "color": color}})

    # Show the unit cell
    view.addUnitCell()

    view.zoomTo()

    # now rotate so we look *down* a or b instead of the default c
    if look.lower() == 'a':
        # default view is along +z (the “c” axis).
        # To look along +x (the “a” axis), rotate –90° around the Y axis:
        view.rotate(-90, 'y')
    elif look.lower() == 'b':
        # to look along +y (the “b” axis), rotate +90° around the X axis:
        view.rotate(-90, 'x')
    # else 'c' or anything else: keep the default

    return view.show()

def animate_view_structure(structure_list, color_map, interval=1.0, look='b'):
    """
    Display an animation in Jupyter by replacing the previous plot with each Structure image.

    Args:
        structure_list (List[pymatgen.Structure]): list of structures to display
        interval (float): seconds between frames
    """
    
    fig, ax = plt.subplots()
    for i, structure in enumerate(structure_list):
        clear_output(wait=True)
        fig = view_structure(structure, color_map, look=look)
        plt.close(fig)
        time.sleep(interval)
    
    return 
