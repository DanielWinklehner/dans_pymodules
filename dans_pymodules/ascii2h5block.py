import h5py
import numpy as np


class TableToH5(object):
    def __init__(self, spacing, r_min, r_max, _output_dir=None, filename='my_h5'):

        self.spacing = spacing
        self.r_min = r_min
        self.r_max = r_max
        self.filename = filename

        # Create a new h5 file
        self.h5_file = h5py.File(filename + '.h5part', )

        # Calculates the size of data arrays
        # noinspection PyTypeChecker
        self._size = np.array((r_max - r_min) / spacing + 1, int)

        # Initialize the h5 data
        # Data Format:
        # Dictionary with keys "ex", "ey", "ez", "hx", "hy", and "hz", which correspond to the vector components
        # of the electric field and the H field.
        self.data = {"ex": np.zeros(self._size),
                     "ey": np.zeros(self._size),
                     "ez": np.zeros(self._size),
                     "hx": np.zeros(self._size),
                     "hy": np.zeros(self._size),
                     "hz": np.zeros(self._size)}

    def set_data(self):
        with open(self.filename + '.table', 'r') as table:
            h = 0
            _s = self._size[0] * self._size[1] * self._size[2]
            _lines = table.readlines()[5:]
            _ex, _ey, _ez = np.zeros(_s), np.zeros(_s), np.zeros(_s)
            # _hx, _hy, _hz = np.zeros(_s), np.zeros(_s), np.zeros(_s)
            for line in _lines:
                _tmp = line.lstrip().rstrip().split()
                _ex[h], _ey[h], _ez[h] = float(_tmp[0]), float(_tmp[1]), float(_tmp[2])
                # _hx, _hy, _hz = float(_tmp[0]), float(_tmp[1]), float(_tmp[2])
                h += 1

        for i in range(self._size[2]):
            for j in range(self._size[1]):
                for k in range(self._size[0]):
                    self.data["ex"][i, j, k] = _ex[i + j * self._size[2] + k * self._size[2] * self._size[1]] * 1e-4
                    self.data["ey"][i, j, k] = _ey[i + j * self._size[2] + k * self._size[2] * self._size[1]] * 1e-4
                    self.data["ez"][i, j, k] = _ez[i + j * self._size[2] + k * self._size[2] * self._size[1]] * 1e-4
                    # self.data["hx"][i, j, k] = _hx[i + j * self._size[2] + k * self._size[2] * self._size[1]] * 1e-3
                    # self.data["hy"][i, j, k] = _hy[i + j * self._size[2] + k * self._size[2] * self._size[1]] * 1e-3
                    # self.data["hz"][i, j, k] = _hz[i + j * self._size[2] + k * self._size[2] * self._size[1]] * 1e-3

    def generate(self):

        # Create the zeroth step and the Block inside of it
        step0 = self.h5_file.create_group("Step#0")
        block = step0.create_group("Block")

        # Create the E Field group
        e_field = block.create_group("Efield")

        # Store the x, y, and z data for the E Field
        e_field.create_dataset("0", data=self.data["ex"])
        e_field.create_dataset("1", data=self.data["ey"])
        e_field.create_dataset("2", data=self.data["ez"])

        # Set the spacing and origin attributes for the E Field group
        e_field.attrs.__setitem__("__Spacing__", self.spacing)
        e_field.attrs.__setitem__("__Origin__", self.r_min)

        # Create the H Field group
        h_field = block.create_group("Hfield")

        # Store the x, y, and z data points for the H Fiend
        h_field.create_dataset("0", data=self.data["hx"])
        h_field.create_dataset("1", data=self.data["hy"])
        h_field.create_dataset("2", data=self.data["hz"])

        # Set the spacing and origin attributes for the H Field group
        h_field.attrs.__setitem__("__Spacing__", self.spacing)
        h_field.attrs.__setitem__("__Origin__", self.r_min)

        # Close the file
        self.h5_file.close()

    def _set_uniform_bfield(self, tesla=None, kgauss=None):
        # Set the magnetic field in the "hz" direction
        if tesla is not None:
            self.data["hz"][:, :, :] = tesla * 10.0
        elif kgauss is not None:
            self.data["hz"][:, :, :] = kgauss


if __name__ == '__main__':
    # Spacing and origin attributes
    spacing = np.array([20.0, 20.0, 20.0])
    r_min = np.array([-100.0, -100.0, -100.0])
    r_max = np.array([100.0, 100.0, 100.0])

    # Assumes that the .table filename is the same as the filename you want to save the h5 to.
    filename = '/home/philip/src/dans_pymodules/dans_pymodules/test_fieldmaps/plate_capacitor_11x11x11_test'

    my_h5 = TableToH5(spacing=spacing, r_min=r_min, r_max=r_max, filename=filename)
    my_h5.set_data()
    # my_h5._set_uniform_bfield(tesla=1.041684)
    my_h5.generate()
