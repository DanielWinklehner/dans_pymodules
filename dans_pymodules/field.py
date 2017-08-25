from scipy.interpolate import RegularGridInterpolator
import gc
# import numpy as np
# import os
from .filedialog import *

__author__ = "Daniel Winklehner"
__doc__ = """Class to load field data save as interpolation functions to get field data at point"""

label_dict = {"EX": "x",
              "EY": "y",
              "EZ": "z",
              "BX": "x",
              "BY": "y",
              "BZ": "z",
              "X": "x",
              "Y": "y",
              "Z": "z"}

dim_numbers = {"x": 0,
               "y": 1,
               "z": 2}


class Field(object):

    def __init__(self,
                 label="Field",
                 dim=0,
                 dim_labels=None,
                 field=None,
                 scaling=1.0,
                 units="m",
                 debug=False):

        self._debug = debug

        self._label = label
        self._dim = dim  # Number of spatial dimensions. Can be 0, 1, 2, 3
        self._dim_labels = dim_labels  # list holding info on which spatial dimensions we are using
        self._filename = None
        self._unit = None
        self._scaling = scaling
        self._field = {"x": None,
                       "y": None,
                       "z": None}

        if field is not None:
            self._field = field

        self._dim_switch = {
            0: self._get_field_0d,
            1: self._get_field_1d,
            2: self._get_field_2d,
            3: self._get_field_3d
        }

        self._units_switch = {
            "m": 1.0,
            "cm": 0.01,
            "mm": 0.001
        }

        assert units in self._units_switch.keys(), "Unit not recognized!"
        self._unit_scale = self._units_switch[units]

    def __call__(self, pts):

        if not isinstance(pts, np.ndarray):
            # print("converting to numpy array")
            pts = np.array(pts)

        if not pts.ndim == 2:
            # print("converting to (1, 3) array")
            pts = np.array([pts])

        # assert len(pts) == 3, "'pts' has to be an array, list, or tuple of length three (x, y, z)!"
        assert self._field is not None, "No field data in memory!"

        return self._dim_switch[self._dim](pts)

    def __str__(self):

        if self._dim == 0:
            return "Field '{}' with {} dimensions and value {}.".format(self._label, self._dim,
                                                                        [item for _, item in self._field.items()])
        else:
            return "Field '{}' with {} dimensions.".format(self._label, self._dim)

    @staticmethod
    def cartesian_to_cylinder(_x, _y, _z):
        """
        Converts polar coordinates into cartesian coordinates.
        :param _x:
        :param _y:
        :param _z:
        :return _r, _t, _z:
        """

        _r = np.sqrt(_x ** 2.0 + _y ** 2.0)
        _t = np.arctan2(_y, _x)

        return _r, _t, _z

    @staticmethod
    def cylinder_to_cartesian(_r, _t, _z):
        """
        Converts polar coordinates into cartesian coordinates.
        :param _r:
        :param _t:
        :param _z:
        :return _x, _y, _z:
        """

        _x = _r * np.cos(_t / 180.0 * np.pi)
        _y = _r * np.sin(_t / 180.0 * np.pi)

        return _x, _y, _z

    def dimensions(self, dim=None):

        if dim is not None:

            assert isinstance(dim, int), "'dim' has to be integer!"
            self._dim = dim

        return self._dim

    def field(self):

        return self._field

    def _get_field_0d(self, pts):

        if len(pts) == 1:

            return self._scaling * np.array([self._field["x"], self._field["y"], self._field["z"]])

        else:

            return self._scaling * np.ones(pts.shape) * np.array([self._field["x"], self._field["y"], self._field["z"]])

    def _get_field_1d(self, pts):

        pts = pts[:, 2]

        if len(pts) == 1:

            return self._scaling * np.array([self._field["x"](pts)[0],
                                             self._field["y"](pts)[0],
                                             self._field["z"](pts)[0]])

        else:

            return self._scaling * self._field["x"](pts), \
                   self._scaling * self._field["y"](pts), \
                   self._scaling * self._field["z"](pts)

    def _get_field_2d(self, pts):

        pts = pts[:, :2]

        if len(pts) == 1:

            return self._scaling * np.array([self._field["x"](pts)[0],
                                             self._field["y"](pts)[0],
                                             self._field["z"](pts)[0]])

        else:

            return self._scaling * self._field["x"](pts), \
                   self._scaling * self._field["y"](pts), \
                   self._scaling * self._field["z"](pts)

    def _get_field_3d(self, pts):

        if len(pts) == 1:

            return self._scaling * np.array([self._field["x"](pts)[0],
                                             self._field["y"](pts)[0],
                                             self._field["z"](pts)[0]])

        else:

            return self._scaling * self._field["x"](pts), \
                   self._scaling * self._field["y"](pts), \
                   self._scaling * self._field["z"](pts)

    def label(self, label=None):

        if label is not None:

            assert isinstance(label, str), "'label' has to be a string!"
            self._label = label

        return self._label

    def load_field_from_file(self, filename=None):

        if filename is None:
            fd = FileDialog()
            filename = fd.get_filename("open")

        assert os.path.exists(filename)
        print("Loading field from file '{}'".format(os.path.split(filename)[1]))
        self._filename = filename

        _, ext = os.path.splitext(filename)

        if ext == ".table":
            print("Detected OPERA table file from extension, loading...")
            self._load_from_table_file(filename)

        return 0

    def _load_from_table_file(self, filename):

        with open(filename, 'r') as infile:

            # Read first line with lengths (first three values)
            _n = np.array([int(val) for val in infile.readline().strip().split()])[:3]

            spatial_dims = 0
            efield_dims = 0
            bfield_dims = 0

            data = {}

            while True:

                line = infile.readline().strip()

                if line == "0":
                    break

                col_no, label, unit = line.split()

                data[label] = {"column": int(col_no) - 1}

                if "LENGU" in unit:
                    data[label]["unit"] = "cm"
                    spatial_dims += 1

                elif "FLUXU" in unit:
                    data[label]["unit"] = "Gauss"
                    bfield_dims += 1

                elif "ELECU" in unit:
                    data[label]["unit"] = "V/cm"
                    efield_dims += 1

                if self._debug:
                    print("Header: '{}' recognized as unit {}".format(label, data[label]["unit"]))

            print("lengths of the spatial dimensions: n = {}".format(_n))

            n = {}
            if "X" in data.keys():
                n["X"] = _n[0]
                if "Y" in data.keys():
                    n["Y"] = _n[1]
                    n["Z"] = _n[2]
                elif "Z" in data.keys():
                    n["Y"] = 1
                    n["Z"] = _n[1]
            elif "Y" in data.keys():
                n["X"] = 1
                n["Y"] = _n[0]
                n["Z"] = _n[1]
            elif "Z" in data.keys():
                n["X"] = 1
                n["Y"] = 1
                n["Z"] = _n[0]

            print("Sorted spatial dimension lengths: {}".format(n))

            # Generate numpy arrays holding the data:
            array_len = n["X"] * n["Y"] * n["Z"]

            # Do some assertions
            self._dim = len(np.where(_n > 1)[0])

            assert self._dim == spatial_dims, "Mismatch between detected number of spatial dimensions and " \
                                              "values in dataset.\n Potentially, only field values are saved " \
                                              "in file. \n This mode will be implemented in the future."

            assert not (efield_dims > 0 and bfield_dims > 0), "Mixed E and B fields are not supported!"

            _data = np.zeros([array_len, spatial_dims + efield_dims + bfield_dims])

            # Read data
            for i, line in enumerate(infile):
                _data[i] = [float(item) for item in line.split()]

        # Get limits and resolution from spatial columns if they exist
        for key, item in data.items():
            if item["unit"] == "cm":
                rmin = _data[0][item["column"]]
                rmax = _data[-1][item["column"]]
                item["limits"] = {"lower": rmin,
                                  "upper": rmax}

                item["dr"] = (rmax - rmin) / (n[key] - 1)

        # TODO: If no spatial columns exist, the user will have to provide the limits!

        if self._debug:
            print("Data Info:")
            for key in sorted(data.keys()):
                print(key, data[key])

        _data = _data.T

        for key, item in data.items():
            if item["unit"] == "cm":
                item["data"] = np.unique(_data[item["column"]])

            else:
                item["data"] = np.reshape(_data[item["column"]], [n["X"], n["Y"], n["Z"]])

        del _data
        gc.collect()

        print("Creating Interpolator for {}D field...".format(self._dim))

        self._dim_labels = []

        # Process the data into the class variables
        # TODO: For now assume that field labels are either "BX", ... or "EX", ...
        label_selector = [["BX", "BY", "BZ"], ["EX", "EY", "EZ"]]
        field_type = len(np.unique([0 for key, item in data.items() if item["unit"] == "V/cm"]))

        for key in label_selector[field_type]:

            if key in data.keys():

                item = data[key]

                self._unit = item["unit"]
                self._dim_labels.append(label_dict[key])

                if self._dim == 1:
                    #  Create a Interpolator object for each of the field dimensions
                    # TODO: For now we assume that column is labeled Z and existed in the file

                    self._field[label_dict[key]] = RegularGridInterpolator(
                        points=[data["Z"]["data"] * self._unit_scale],
                        values=item["data"].squeeze(),
                        bounds_error=False,
                        fill_value=0.0)
                elif self._dim == 2:
                    #  Create a Interpolator object for each of the field dimensions
                    # TODO: For now we assume that columns are labeled X, Y and existed in the file

                    self._field[label_dict[key]] = RegularGridInterpolator(
                        points=[data["X"]["data"] * self._unit_scale,
                                data["Y"]["data"] * self._unit_scale],
                        values=item["data"].squeeze(),
                        bounds_error=False,
                        fill_value=0.0)

                elif self._dim == 3:
                    #  Create a Interpolator object for each of the field dimensions
                    # TODO: For now we assume that columns are labeled X, Y, Z and existed in the file

                    self._field[label_dict[key]] = RegularGridInterpolator(
                        points=[data["X"]["data"] * self._unit_scale,
                                data["Y"]["data"] * self._unit_scale,
                                data["Z"][
                                    "data"] * self._unit_scale],
                        values=item["data"],
                        bounds_error=False,
                        fill_value=0.0)
            else:

                self._dim_labels.append(label_dict[key])

                if self._dim == 1:

                    _z = np.array([-1e20, 1e20])
                    _f = np.zeros(2)

                    self._field[label_dict[key]] = RegularGridInterpolator(points=[_z],
                                                                           values=_f,
                                                                           bounds_error=False,
                                                                           fill_value=0.0)

                elif self._dim == 2:

                    _x = np.array([-1e20, 1e20])
                    _y = np.array([-1e20, 1e20])
                    _f = np.zeros([2, 2])

                    self._field[label_dict[key]] = RegularGridInterpolator(points=[_x, _y],
                                                                           values=_f,
                                                                           bounds_error=False,
                                                                           fill_value=0.0)

                elif self._dim == 3:

                    _x = np.array([-1e20, 1e20])
                    _y = np.array([-1e20, 1e20])
                    _z = np.array([-1e20, 1e20])
                    _f = np.zeros([2, 2, 2])

                    self._field[label_dict[key]] = RegularGridInterpolator(points=[_x, _y, _z],
                                                                           values=_f,
                                                                           bounds_error=False,
                                                                           fill_value=0.0)

            # save_pickle_fn = os.path.join(os.path.split(filename)[0], "temp_pickle.dat")
            # import pickle
            # pickle.dump(data, save_pickle_fn)

        return 0

    def scaling(self, scaling=None):

        if scaling is not None:

            assert isinstance(scaling, float), "'scaling factor' has to be float!"
            self._scaling = scaling

        return self._scaling

# Set up some tests
if __name__ == "__main__":
    mydebug = True
    import matplotlib.pyplot as plt
    # if platform.node() == "Mailuefterl":
    #     folder = r"D:\Daniel\Dropbox (MIT)\Projects" \
    #              r"\RFQ Direct Injection\Cyclotron"
    # elif platform.node() == "TARDIS":
    #     folder = r"D:\Dropbox (MIT)\Projects" \
    #              r"\RFQ Direct Injection\Cycloton"
    # else:
    #     folder = r"C:\Users\Daniel Winklehner\Dropbox (MIT)\Projects" \
    #              r"\RFQ Direct Injection\Cyclotron"

    folder = "D:\Daniel\Dropbox(MIT)\Projects\IsoDAR\Ion Source\MIST - 1_Magnet"

    # Test manual 0D field creation (constant everywhere)
    # bfield = Field(label="Constant Cyclotron B-Field",
    #                dim=0,
    #                field={'x': 0, 'y': 0, 'z': 1.04},
    #                debug=mydebug)
    #
    # print(bfield)
    # print(bfield(np.array([0.0, 0.0, 0.0])))

    # Test OPERA 1D field import
    bfield1 = Field(label="Test Cyclotron B-Field 1D",
                    debug=mydebug,
                    scaling=-1.0)
    bfield1.load_field_from_file(
        filename=os.path.join(folder, "BZ_vs_Z_along_z_axis.table"))

    print(bfield1)
    print(bfield1(np.array([0.0, 0.0, 0.0])))

    x = np.zeros(2000)
    y = np.zeros(2000)
    z = np.linspace(-15, 5, 2000)

    points = np.vstack([x, y, z]).T

    _, _, bz = bfield1(points)

    plt.plot(z, bz)
    plt.show()

    # Test OPERA 2D field import
    # bfield = Field(label="Test Cyclotron B-Field 2D",
    #                debug=mydebug)
    # bfield.load_field_from_file(
    #     filename=os.path.join(folder, "Bz_of_x_and_y.table"))
    #
    # print(bfield)
    # print(bfield(np.array([0.0, 0.0, 0.0])))
    #
    # x = np.linspace(-10, 10, 100)
    # y = np.linspace(-10, 10, 100)
    #
    # mesh_x, mesh_y = meshgrid(x, y, indexing='ij')
    # points = np.vstack([mesh_x.flatten(), mesh_y.flatten(), np.zeros(10000)]).T
    #
    # _, _, bz = bfield(points)
    #
    # plt.contour(x, y, bz.reshape([100, 100]), 40)
    # plt.colorbar()
    # plt.show()

    # Test OPERA 3D field import
    # bfield = Field(label="Test Cyclotron B-Field 3D",
    #                debug=mydebug,
    #                scaling=-1.0)
    # bfield.load_field_from_file(
    #     filename=os.path.join(folder, "Bx_By_Bz_of_x_y_z.table"))
    #
    # print(bfield)
    # print(bfield(np.array([0.0, 0.0, 0.0])))

    # x = np.linspace(-10, 10, 100)
    # y = np.linspace(-10, 10, 100)
    #
    # from scipy import meshgrid
    # mesh_x, mesh_y = meshgrid(x, y, indexing='ij')
    # points = np.vstack([mesh_x.flatten(), mesh_y.flatten(), np.zeros(10000)]).T
    #
    # _, _, bz = bfield(points)
    #
    # plt.contour(x, y, bz.reshape([100, 100]), 40)
    # plt.colorbar()
    # plt.show()

    # x = np.zeros(200)
    # y = np.zeros(200)
    # z = np.linspace(-15, 5, 200)
    #
    # points = np.vstack([x, y, z]).T
    #
    # _, _, bz = bfield(points)
    #
    # plt.plot(z, bz)
    # plt.show()
