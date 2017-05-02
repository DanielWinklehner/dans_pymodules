import h5py
import numpy as np


class Ascii2H5(object):
    def __init__(self, filename, spacing, r_min, r_max, _output_dir=None):
        self.spacing = spacing
        self.r_min = r_min
        self.r_max = r_max

        # TODO: Add path to filename
        # Create a new h5 file

        self.h5_file = h5py.File(filename, )
        # h5_file = h5py.File("C:/Users/Daniel/Desktop/Test.h5", )

        # Calculates the size of data arrays
        _size = np.array((r_max - r_min) / spacing + 1, int)

        # Data Format:
        # Dictionary with keys "ex", "ey", "ez", "hx", "hy", and "hz", which correspond to the vector components
        # of the electric field and the H field.
        self.data = {"ex": np.zeros(_size),
                     "ey": np.zeros(_size),
                     "ez": np.zeros(_size),
                     "hx": np.zeros(_size),
                     "hy": np.zeros(_size),
                     "hz": np.zeros(_size)}

    def set_data(self, data):
        self.data = data

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

if __name__ == '__main__':
    spacing = np.array([1.0, 1.0, 1.0])
    r_min = np.array([-80.0, -80.0, -100.0])
    r_max = np.array([80.0, 80.0, 50.0])
    filename='my_h5.h5'

    my_h5 = Ascii2H5(filename=filename, spacing=spacing, r_min=r_min, r_max=r_max)
    my_h5.generate()

