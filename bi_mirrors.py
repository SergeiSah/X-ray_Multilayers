import pandas as pd
from scipy.optimize import fsolve
from compounds import Compound, HC_CONST
import numpy as np


class BiMirror:

    def __init__(self, absorber: Compound, spacer: Compound):
        self.factors = absorber.factors.merge(spacer.factors, on='energy', how='outer',
                                              suffixes=('_a', '_s')).sort_index().interpolate()

        # intermediate settlements
        self.__f = (self.factors.delta_a - self.factors.delta_s) / (self.factors.beta_a - self.factors.beta_s)
        self.__g = self.factors.beta_s / (self.factors.beta_a - self.factors.beta_s)

    def calc_optimal_gamma(self):
        df = pd.DataFrame([0] * self.factors.shape[0], columns=['gamma'], index=self.factors.index)

        for i in range(self.factors.shape[0]):
            df.iloc[i] = fsolve(self._optimal_gamma, np.array([0.4999]), (self.__g.iloc[i],))

        # After absorption edges gamma becomes negative. All negative values (x) transformed to '1 + x'
        df[df.gamma < 0] = df[df.gamma < 0].apply(lambda x: 1 + x)

        return df.gamma.astype(np.float64)

    def calc_max_reflection(self,
                            polarization='circ',
                            normal_angle=0,
                            gamma=None,
                            names=None) -> pd.DataFrame:

        opt_constants = self.factors[['delta_a', 'beta_a', 'delta_s', 'beta_s']].copy()

        gamma = self.__process_gamma_input(gamma)
        s_pol = self._polarization_func('s', normal_angle)
        p_pol = self._polarization_func('p', normal_angle)

        y = np.pi * (gamma + self.__g) / np.sin(np.pi * gamma)
        u_s = np.sqrt((1 - np.power(s_pol / y, 2)) / (1 + np.power(self.__f * s_pol / y, 2)))
        u_p = np.sqrt((1 - np.power(p_pol / y, 2)) / (1 + np.power(self.__f * p_pol / y, 2)))

        r_s_max = (1 - u_s) / (1 + u_s)
        r_p_max = (1 - u_p) / (1 + u_p)

        if polarization.lower() == 's':
            opt_constants['R_S_max'] = r_s_max
        elif polarization.lower() == 'p':
            opt_constants['R_P_max'] = r_p_max
        elif polarization.lower() in ['c', 'circ', 'circular']:
            opt_constants.loc[:, 'R_circ_max'] = (r_s_max + r_p_max) / 2
        else:
            opt_constants['R_S_max'] = r_s_max
            opt_constants['R_P_max'] = r_p_max
            opt_constants['R_circ_max'] = (r_s_max + r_p_max) / 2

        if names is not None:
            opt_constants.columns = list(map(lambda k: k.replace('_a', f'_{names[0]}').replace('_s', f'_{names[1]}'), opt_constants.columns))

        return opt_constants.iloc[:, 4:]

    @staticmethod
    def _optimal_gamma(gamma, g):
        return np.tan(np.pi * gamma) - np.pi * gamma - np.pi * g

    @staticmethod
    def _polarization_func(pol: str, angle: float):
        if pol.lower() == 's':
            return 1.0
        elif pol.lower() == 'p':
            ang_rad = angle / 180 * np.pi
            return np.cos(2 * ang_rad)
        else:
            raise TypeError('Wrong polarization type. Must be "s" or "p".')

    def __calc_y(self, gamma):
        return np.pi * (gamma + self.__g) / np.sin(np.pi * gamma)

    def __process_gamma_input(self, gamma):
        if gamma is None:
            gamma = self.calc_optimal_gamma()
        elif isinstance(gamma, str):
            if gamma.lower() in ['o', 'opt', 'optimal']:
                gamma = self.calc_optimal_gamma()
            else:
                raise ValueError(f'Wrong gamma mode. Must be "o" or "opt" or "optimal"')
        elif isinstance(gamma, float) and .0 <= gamma <= 1.0:
            gamma = gamma
        else:
            raise TypeError(f'Wrong gamma. Must be float in the range [0, 1] or "optimal"')
        return gamma

    def calculate_resolving_power(self,
                                  polarization: str = 'circ',
                                  normal_angle: float = .0,
                                  gamma: float = None) -> pd.DataFrame:

        gamma = self.__process_gamma_input(gamma)
        ang_rad = normal_angle / 180 * np.pi
        polarization = self._polarization_func(polarization, normal_angle)

        opt_constants = self.factors[['delta_a', 'beta_a', 'delta_s', 'beta_s']]
        y = self.__calc_y(gamma)

        c = 2 * np.sqrt(2) / 3 * np.pi
        im_mu = gamma * opt_constants.beta_a + (1 - gamma) * opt_constants.beta_s

        opt_constants['RP'] = pd.NA
        for i, en in enumerate(opt_constants.index.values):
            xi = abs(self.__f.iloc[i] * polarization) / y.iloc[i]
            if xi >= 10:
                opt_constants.iloc[i, 5] = c / np.sin(np.pi * opt_constants.loc[en, 'gamma']) * \
                                           np.power(np.cos(ang_rad), 2) / abs(polarization) / \
                                           (opt_constants.loc[en, 'delta_a'] - opt_constants.loc[en, 'delta_s'])
            else:
                opt_constants.iloc[i, 5] = np.power(np.cos(ang_rad), 2) / im_mu.iloc[i] / \
                                     np.power((1 - np.power(polarization / y.iloc[i], 2)) *
                                              (1 + np.power(self.__f.iloc[i] * polarization / y.iloc[i], 2)), 0.75)
        return opt_constants['RP']

    def calculate_penetration_depth(self, normal_angle: float, gamma: float = None) -> pd.DataFrame:
        gamma = self.__process_gamma_input(gamma)
        opt_constants = self.factors[['delta_a', 'beta_a', 'delta_s', 'beta_s']]
        s_pol = self._polarization_func('s', normal_angle)
        y = self.__calc_y(gamma)

        wavelengths = pd.DataFrame(HC_CONST / opt_constants.index.values, columns=['wv']).set_index(opt_constants.index)
        im_mu = gamma * opt_constants.beta_a + (1 - gamma) * opt_constants.beta_s

        return wavelengths.wv * np.cos(normal_angle) / np.pi / im_mu / np.sqrt(
            (1 - np.power(s_pol / y, 2)) * (1 + np.power(self.__f * s_pol / y, 2)))


if __name__ == '__main__':
    mirror = BiMirror(Compound('Mo'), Compound('Be'))
    print(mirror.calc_max_reflection())
