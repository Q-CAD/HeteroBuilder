import py3Dmol
from pymatgen.core.periodic_table import Element 

def view_structure(structure, color_map):
    view = py3Dmol.view(width=800, height=800)
    cif_str = structure.to(fmt="cif")
    view.addModel(cif_str, "cif")

    # Apply sphere style and color for each element
    for element, color in color_map.items():
        view.setStyle({"elem": element}, {"sphere": {"radius": Element(element).atomic_radius, "color": color}})

    # Show the unit cell
    view.addUnitCell()

    view.zoomTo()
    return view.show()
