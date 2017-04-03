import numpy as np

__author__ = "Daniel Winklehner"
__doc__ = "Simple 2D vector class covering basic vector operations."


class Vector(object):
    """
    Simple 2D vector class covering basic vector operations
    """

    def __init__(self, vector=np.ones(2, 'd'), p0=None, p1=None):
        """
        Constructor
        """
        if p0 is not None and p1 is not None:
            assert len(p0) == 2 and len(p1) == 2
            self.vector = np.array(p1, 'd') - np.array(p0, 'd')
        else:
            assert len(vector) == 2
            self.vector = np.array(vector, 'd')

        self.length = self.get_length()

    def __getitem__(self, item):
        return self.vector[item]

    def __str__(self):
        return str(self.vector)

    def __mul__(self, other):
        try:
            v2 = other.vector
            return self[0] * v2[0] + self[1] * v2[1]

        except AttributeError:

            return Vector(vector=(self.vector * other))

    def __rmul__(self, other):
        try:
            v2 = other.vector
            return self[0] * v2[0] + self[1] * v2[1]

        except AttributeError:

            return Vector(vector=(self.vector * other))

    def __div__(self, other):
        return Vector(vector=(self.vector / other))

    def __rdiv__(self, other):
        return Vector(vector=(other / self.vector))

    def __add__(self, other):
        try:

            return Vector(vector=(self.vector + other.vector))

        except AttributeError:

            print("Can only add and subtract vectors from vectors!")

    def __radd__(self, other):
        try:

            return Vector(vector=(self.vector + other.vector))

        except AttributeError:

            print ("Can only add and subtract vectors from vectors!")

    def __sub__(self, other):
        try:

            return Vector(vector=(self.vector - other.vector))

        except AttributeError:

            print("Can only add and subtract vectors from vectors!")

    def __rsub__(self, other):
        try:

            return Vector(vector=(other.vector - self.vector))

        except AttributeError:

            print ("Can only add and subtract vectors from vectors!")

    def get_length(self):
        return np.sqrt(self.vector[0] ** 2.0 + self.vector[1] ** 2.0)

    def normalize(self):

        return Vector(vector=(self.vector / self.length))

    def rotate_ccw(self):

        return Vector(vector=np.array([-self.vector[1], self.vector[0]]))

    def rotate_cw(self):

        return Vector(vector=np.array([self.vector[1], -self.vector[0]]))

    def angle(self, v2):
        """
        Calculate the angle from the present vector to the given second_vector
        """
        return np.arccos(self * v2 / self.get_length() / v2.get_length())
