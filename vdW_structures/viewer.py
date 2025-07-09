import py3Dmol
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
