from scipy.interpolate import RegularGridInterpolator
import gc
from scipy.interpolate import interp1d, interp2d, griddata
import numpy as np
import os
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

        _r = np.sqrt(_x * _x + _y * _y)
        _t = np.arctan2(_y, _x)

        return _r, _t, _z

    @staticmethod
    def cylinder_to_cartesian_field(_r, _t, _z, _br, _bt, _bz):
        """
        Converts polar coordinates into cartesian coordinates.
        :param _r:
        :param _t:
        :param _z:
        :param _br:
        :param _bt:
        :param _bz:
        :return _x, _y, _z:
        """

        _t_rad = np.deg2rad(_t)

        _x = _r * np.cos(_t_rad)
        _y = _r * np.sin(_t_rad)

        _bx = _br * np.cos(_t_rad) - _bt * np.sin(_t_rad)
        _by = _br * np.sin(_t_rad) + _bt * np.cos(_t_rad)

        return _x, _y, _z, _bx, _by, _bz

    @staticmethod
    def cylinder_to_cartesian(_r, _t, _z):
        """
        Converts polar coordinates into cartesian coordinates.
        :param _r:
        :param _t:
        :param _z:
        :return _x, _y, _z:
        """

        _x = _r * np.cos(np.deg2rad(_t))
        _y = _r * np.sin(np.deg2rad(_t))

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

            return self._scaling * np.array([self._field["x"],
                                             self._field["y"],
                                             self._field["z"]])

        else:

            return self._scaling * self._field["x"] * np.ones(len(pts)), \
                   self._scaling * self._field["y"] * np.ones(len(pts)), \
                   self._scaling * self._field["z"] * np.ones(len(pts))

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

        if ext == ".map":
            print("Detected AIMA Agora field file. All '*.map' files in the current folder will be used!")
            self._load_from_agora_file(filename)

        if ext == ".table":
            print("Detected OPERA table file from extension, loading...")
            self._load_from_table_file(filename)

        if ext == ".comsol":
            print("Detected COMSOL file from extension, loading...")
            self._load_from_comsol_file(filename)

        return 0

    def _load_from_agora_file(self, filename, mirror=False):

        self._dim = 3
        interp_method = "linear"

        # Find all *.map files in the current folder
        _path = os.path.split(filename)[0]
        _all_files = []
        for entry in os.scandir(_path):
            if entry.is_file() and os.path.splitext(entry.name)[1] == ".map":
                _all_files.append(entry.name)

        _br = np.zeros([])
        _bf = np.zeros([])
        _bz = np.zeros([])

        _r = np.zeros([])
        _th = np.zeros([])
        _z = np.zeros([])

        _metadata = []
        _metadata_types = np.dtype([("fn", np.unicode_, 1024),   # filename
                                    ("fcomp", np.unicode_, 2),   # field component (bf, br, bz)
                                    ("zpos", float)])            # z position (mm)

        for _file in _all_files:
            _z_pos, res = _file.split("Z")[1].split("-")  # mm
            _b_comp = res.split(".")[0]

            if self._debug:
                print("File '{}' with component '{}' has Z position {} mm".format(_file, _b_comp, _z_pos))

            _metadata.append((os.path.join(_path, _file), _b_comp, _z_pos))

        _metadata = np.array(_metadata, dtype=_metadata_types)

        # Assert that there are an equal number of files for each component and each z position
        assert \
            np.array_equal(_metadata[np.where(_metadata["fcomp"] == "bz")]["zpos"],
                           _metadata[np.where(_metadata["fcomp"] == "br")]["zpos"]) and \
            np.array_equal(_metadata[np.where(_metadata["fcomp"] == "bz")]["zpos"],
                           _metadata[np.where(_metadata["fcomp"] == "bf")]["zpos"]), \
            "Not all components have the same z positions. Maybe a file is missing?"

        # Assert that r and theta resolution and limits are the same
        with open(_metadata["fn"][0]) as infile:
            infile.readline()
            _nth, _nr = [int(value) for value in infile.readline().strip().split()]
            _sr, _dr = [float(value) for value in infile.readline().strip().split()]
            _sth, _dth = [float(value) for value in infile.readline().strip().split()]
            old_limits = np.array([_nth, _nr, _sr, _dr, _sth, _dth])
        for _filename in _metadata["fn"][1:]:
            with open(_filename, 'r') as infile:
                infile.readline()
                _nth, _nr = [int(value) for value in infile.readline().strip().split()]
                _sr, _dr = [float(value) for value in infile.readline().strip().split()]
                _sth, _dth = [float(value) for value in infile.readline().strip().split()]
                new_limits = np.array([_nth, _nr, _sr, _dr, _sth, _dth])
                assert np.array_equal(old_limits, new_limits), "Limits for r, th in some of the files are not the same!"
                old_limits = new_limits

        _r_unique = np.round(np.linspace(_sr, _nr * _dr, _nr, endpoint=False), 10)  # cm
        _th_unique = np.round(np.linspace(_sth, _nth * _dth, _nth, endpoint=False), 10)  # degree
        _z_unique = np.sort(np.unique(_metadata["zpos"])) / 10.0  # mm --> cm
        _n_z_pos = len(_z_unique)

        _r, _th = np.meshgrid(_r_unique, _th_unique)

        # _z = np.tile(_z_unique, (_r.size, 1)).T.flatten()
        # _r = np.tile(_r.flatten(), (_n_z_pos, 1)).flatten()
        # _th = np.tile(_th.flatten(), (_n_z_pos, 1)).flatten()

        # _data = {"r": _r,
        #          "th": _th,
        #          "z": _z,
        #          "br": np.array([], float),
        #          "bf": np.array([], float),
        #          "bz": np.array([], float)}

        for _z in _z_unique:
            _data = {}
            for _val in _metadata[np.where(_metadata["zpos"] == _z)]:
                _filename, _fcomp, _zpos = _val

                with open(_filename, 'r') as infile:
                    _raw_data = infile.readlines()[4:-1]

                _data[_fcomp] = np.array([np.fromstring(_line, sep=" ") for _line in _raw_data]).flatten()

            _data["x"], _data["y"], _data["z"], _data["bx"], _data["by"], _data["bz"] = \
                self.cylinder_to_cartesian_field(_r.flatten(), _th.flatten(), _z, _data["br"], _data["bf"], _data["bz"])

            # Naturally, this is not a regular grid in x, y now...
            print()
            print("Starting griddata... "),

            # grid_x, grid_y = np.mgrid[self.limits[0]:self.limits[1]:500j, self.limits[2]:self.limits[3]:500j]

            _x = np.round(np.linspace(min(_data["x"]), max(_data["x"]), 1000), 10)
            _y = np.round(np.linspace(min(_data["y"]), max(_data["y"]), 1000), 10)

            grid_x, grid_y = np.meshgrid(_x, _y, indexing='ij')
            grid_z = np.ones(grid_x.shape) * _z

            bx = griddata((_data["x"], _data["y"]), _data["bx"],
                          (grid_x, grid_y),
                          method=interp_method,
                          fill_value=0.0)

            by = griddata((_data["x"], _data["y"]), _data["by"],
                          (grid_x, grid_y),
                          method=interp_method,
                          fill_value=0.0)

            bz = griddata((_data["x"], _data["y"]), _data["bz"],
                          (grid_x, grid_y),
                          method=interp_method,
                          fill_value=0.0)

            # from matplotlib import pyplot as plt
            # plt.subplot(221)
            # plt.title("Bx")
            # plt.xlabel("x (cm)")
            # plt.ylabel("y (cm)")
            # cont = plt.contourf(grid_x, grid_y, bx)
            # plt.colorbar(cont)
            # plt.subplot(222)
            # plt.title("By")
            # plt.xlabel("x (cm)")
            # plt.ylabel("y (cm)")
            # cont = plt.contourf(grid_x, grid_y, by)
            # plt.colorbar(cont)
            # plt.subplot(223)
            # plt.title("Bz")
            # plt.xlabel("x (cm)")
            # plt.ylabel("y (cm)")
            # cont = plt.contourf(grid_x, grid_y, bz)
            # plt.colorbar(cont)
            # plt.show()

            print("Done!")
        #
        # # Transform into cartesian coordinates before creating interpolator object.
        # # TODO: All of this should be restructured to allow for cylindrical fields, optionally with symmetries
        # _data["x"], _data["y"], _data["z"], _data["bx"], _data["by"], _data["bz"] = \
        #     self.cylinder_to_cartesian_field(_data["r"], _data["th"], _data["z"], _data["br"], _data["bf"], _data["bz"])
        #
        # for key in ["bx", "by", "bz"]:
        #     self._field[label_dict[key]] = RegularGridInterpolator(
        #         points=[_data["x"] * self._unit_scale,
        #                 _data["y"] * self._unit_scale,
        #                 _data["z"] * self._unit_scale],
        #         values=_data[key],
        #         bounds_error=False,
        #         fill_value=0.0)

        return 0

    def _load_from_comsol_file(self, filename):

        self._dim = 3

        label_map = {"mir1x": "X",
                     "mir1y": "Y",
                     "mir1z": "Z",
                     "mf.Bx": "BX",
                     "mf.By": "BY",
                     "mf.Bz": "BZ"}

        with open(filename, 'r') as infile:

            nlines = 9  # Number of header lines
            data = {}

            for i in range(nlines):
                line = infile.readline().strip()
                sline = line.split()
                if i == 3:
                    # self._dim = int(sline[2])
                    spatial_dims = 3
                    efield_dims = 0
                    bfield_dims = 3
                elif i == 4:
                    array_len = int(sline[2])
                elif i == 7:
                    l_unit = sline[3]
                elif i == 8:
                    if "(T)" in sline:
                        b_unit = "T"
                    j = 0
                    for label in sline:
                        if label not in ["%", "(T)"]:
                            nlabel = label_map[label]
                            data[nlabel] = {"column": j}
                            if nlabel in ["X", "Y", "Z"]:
                                data[nlabel]["unit"] = l_unit
                            elif nlabel in ["BX", "BY", "BZ"]:
                                data[nlabel]["unit"] = b_unit
                            # Another statement for EX, EY, EZ
                            if self._debug:
                                print("Header: '{}' recognized as unit {}".format(nlabel, data[nlabel]["unit"]))
                            j += 1

            _data = np.zeros([array_len, spatial_dims + efield_dims + bfield_dims])

            # Read data
            for i, line in enumerate(infile):
                _data[i] = [float(item) for item in line.split()]

        if self._debug:
            print("Data Info:")
            for key in sorted(data.keys()):
                print(key, data[key])

        _data = _data[np.lexsort(_data.T[[1]])]
        _data = _data[np.lexsort(_data.T[[0]])].T

        n = {}
        n["X"], n["Y"], n["Z"] = len(np.unique(_data[0])), \
                                 len(np.unique(_data[1])), \
                                 len(np.unique(_data[2]))

        for key, item in data.items():
            if item["unit"] == "mm":
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

                if self._dim == 3:
                    #  Create a Interpolator object for each of the field dimensions
                    # TODO: For now we assume that columns are labeled X, Y, Z and existed in the file

                    self._field[label_dict[key]] = RegularGridInterpolator(
                        points=[data["X"]["data"] * self._unit_scale,
                                data["Y"]["data"] * self._unit_scale,
                                data["Z"]["data"] * self._unit_scale],
                        values=item["data"],
                        bounds_error=False,
                        fill_value=0.0)

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
                # print(label)
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
                print(item["data"].shape)
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
    bfield1 = Field(label="Test importing AIMA field 3D",
                    debug=mydebug)
    bfield1.load_field_from_file()
    exit()
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

    import matplotlib.pyplot as plt
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
