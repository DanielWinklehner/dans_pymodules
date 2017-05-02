import h5py
import numpy as np

spacing = np.array([1.0, 1.0, 1.0])
rmin = np.array([-80.0, -80.0, -100.0])
rmax = np.array([80.0, 80.0, 50.0])

size = np.array((rmax - rmin) / spacing + 1, int)
print(size)

data = {"ex": np.zeros(size),
        "ey": np.zeros(size),
        "ez": np.zeros(size),
        "hx": np.zeros(size),
        "hy": np.zeros(size),
        "hz": np.zeros(size)
        }

h5_file = h5py.File("C:/Users/Daniel/Desktop/Test.h5", )

step0 = h5_file.create_group("Step#0")
block = step0.create_group("Block")

efield = block.create_group("Efield")
efield_x = efield.create_dataset("0", data=data["ex"])
efield_y = efield.create_dataset("1", data=data["ey"])
efield_z = efield.create_dataset("2", data=data["ez"])

efield.attrs.__setitem__("__Spacing__", spacing)
efield.attrs.__setitem__("__Origin__", rmin)

hfield = block.create_group("Hfield")
hfield_x = hfield.create_dataset("0", data=data["hx"])
hfield_y = hfield.create_dataset("1", data=data["hy"])
hfield_z = hfield.create_dataset("2", data=data["hz"])

hfield.attrs.__setitem__("__Spacing__", spacing)
hfield.attrs.__setitem__("__Origin__", rmin)

h5_file.close()
