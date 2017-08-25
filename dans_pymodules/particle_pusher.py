import sys
import numpy as np
if sys.version_info.major == 3:
    from .particles import IonSpecies
    from .field import Field
else:
    from particles import IonSpecies
    from field import Field


class ParticlePusher(object):

    def __init__(self, ion, algorithm="boris"):

        self._alg_switch = {
            "boris": self._v_boris,
            "leapfrog": self._v_leapfrog,
            "tajima_implicit": self._v_tajima_implicit
        }

        assert isinstance(ion, IonSpecies), "'ion' needs to be an IonSpecies object!"

        self._ion = ion

        algorithm = algorithm.lower()

        assert algorithm in self._alg_switch.keys(), "'{}' is not a known algorithm!".format(algorithm)

        self._algorithm = algorithm

        self._v = self._alg_switch[self._algorithm]

    def algorithm(self, algorithm=None):

        if algorithm in self._alg_switch.keys():
            self._algorithm = algorithm
            self._v = self._alg_switch[self._algorithm]
        else:
            print("'{}' is not a known algorithm, keeping current algorithm ('{}')".format(algorithm, self._algorithm))

        return self._algorithm

    def _v_boris(self, _v, _efield, _bfield, _dt):

        t = 0.5 * self._ion.q_over_m() * _bfield * _dt
        s = 2.0 * t / (1.0 + np.linalg.norm(t) ** 2.0)
        v_minus = _v + 0.5 * self._ion.q_over_m() * _efield * _dt
        v_prime = v_minus + np.cross(v_minus, t)
        v_plus = v_minus + np.cross(v_prime, s)

        return v_plus + 0.5 * self._ion.q_over_m() * _efield * _dt

    def _v_leapfrog(self, _v, _efield, _bfield, _dt):

        return _v + self._ion.q_over_m() * (_efield + np.cross(_v, _bfield)) * _dt

    def _v_tajima_implicit(self, _v, _efield, _bfield, _dt):

        b_norm = np.linalg.norm(_bfield)

        rot_mat = (1.0 / b_norm) * np.array([[0.0, _bfield[2], -_bfield[1]],
                                             [-_bfield[2], 0.0, _bfield[0]],
                                             [-_bfield[1], -_bfield[0], 0.0]])

        epsilon = 0.5 * self._ion.q_over_m() * b_norm * _dt
        mat_p = np.eye(3) + epsilon * rot_mat
        mat_m_inv = np.linalg.inv(np.eye(3) - epsilon * rot_mat)
        _v = np.matmul(np.matmul(mat_m_inv, mat_p), _v) + np.matmul(mat_m_inv, _efield) * self._ion.q_over_m() * _dt

        return _v

    def push(self, _r, _v, _efield, _bfield, _dt):

        _v = self._v(_v, _efield, _bfield, _dt)  # Call the velocity function determined by the algorithm
        _r += _v * _dt

        return _r, _v


if __name__ == "__main__":
    h2p = IonSpecies(name="H2_1+", energy_mev=1.0, label="$\mathrm{H}_2^+$")
    h2p.calculate_from_energy_mev(0.07 / h2p.a())
    print("Cyclotron radius should be {} m".format(h2p.b_rho()))

    pusher = ParticlePusher(h2p, "boris")
    efield1 = Field(dim=0, field={"x": 0.0, "y": 0.0, "z": 0.0})
    bfield1 = Field(dim=0, field={"x": 0.0, "y": 0.0, "z": -1.0})

    nsteps = 10000
    dt = 1e-10  # (1 ps --> s)

    r = np.zeros([nsteps + 1, 3])
    v = np.zeros([nsteps + 1, 3])
    r[0] = [h2p.b_rho(), 0.0, 0.0]
    v[0] = [0.0, h2p.v_m_per_s(), 0.0]

    # initialize the velocity half a step back:
    ef = efield1(r[0])
    bf = bfield1(r[0])
    _, v[0] = pusher.push(r[0], v[0], ef, bf, -0.5 * dt)

    for i in range(nsteps):

        ef = efield1(r[i])
        bf = bfield1(r[i])

        r[i + 1], v[i + 1] = pusher.push(r[i], v[i], ef, bf, dt)

    import matplotlib.pyplot as plt

    plt.plot(r[:, 0], r[:, 1])
    plt.gca().set_aspect('equal')
    plt.show()
