import sys

if sys.version_info >= (3, 2):

    from .particles import *
    from .mycolors import *
    from .pylatex import *
    from .particle_distribution import *
    from .vector2d import *
    from .read_igun import *
    from .power_of_two import power_of_two
    from .coordinate_transformation_3d import *

    try:
        from .filedialog import *
        from .label_combo import *
        from .mpl_canvas_wrapper import *
    except ModuleNotFoundError:
        print("GObject Intropection not found, filedialog, label_combo, and mpl_canvas_wrapper not available!")

elif sys.version_info == (2, 7):

    from particles import *
    from mycolors import *
    from filedialog import *
    from pylatex import *
    from particle_distribution import *
    from label_combo import *
    from mpl_canvas_wrapper import *
    from vector2d import *
    from read_igun import *
    from power_of_two import power_of_two
    from coordinate_transformation_3d import *
