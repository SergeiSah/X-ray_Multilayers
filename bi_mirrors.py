"""
https://doi.org/10.1007/BF02581074
I.V. Kozhevnikov, A.V. Vinogradov, "Multilayer X-ray mirrors",
Journal of Russian Laser Research, vol. 16, no. 4, 1995

"BiMirror" is a class for calculating the reflectivity, resolving power, penetration depth
of an X-ray mirror, comprised of two materials, using formulas from Kozhevnikov's work.

Only Bragg peak with n=1 is considered.

Definitions:
l_a and l_b - thicknesses of layers;
l = l_a + l_b - the structure period;
N is number of pairs of layers (periods) in the structure;
L = Nl - the structure thickness;
gamma = l_a / l - the fraction of period occupied by the absorption layer;
mu = gamma * epsilon_a + (1 - gamma) * epsilon_b - the permittivity averaged over the period;
"""

import pandas as pd
from scipy.optimize import fsolve
from compounds import Compound, HC_CONST
import numpy as np


class BiMirror:

    def __init__(self, absorber: Compound, spacer: Compound):
        self.mirror = f'{absorber.chem_formula}/{spacer.chem_formula}'
        self.absorber = absorber
        self.spacer = spacer

        # intermediate settlements
        self.__f = None
        self.__g = None

    @property
    def permittivity(self):
        epsilon = self.absorber.permittivity.join(self.spacer.permittivity, how='outer',
                                                  lsuffix='_a', rsuffix='_s').interpolate('index')
        self.__f = (epsilon.e1_a - epsilon.e1_s) / (epsilon.e2_a - epsilon.e2_s)
        self.__g = epsilon.e2_s / (epsilon.e2_a - epsilon.e2_s)
        return epsilon

    def calc_optimal_gamma(self):
        e = self.permittivity
        df = pd.DataFrame([0] * e.shape[0], columns=['gamma'], index=e.index)

        for i in range(e.shape[0]):
            df.iloc[i] = fsolve(self._optimal_gamma, np.array([0.4999]), (self.__g.iloc[i],))

        # After absorption edges gamma becomes negative. All negative values (x) transformed to '1 + x'
        # FIXME: this is not correct, but it works for now
        df[df.gamma < 0] = df[df.gamma < 0].apply(lambda x: 1 + x)

        return df.gamma.astype(np.float64)

    def calc_max_reflection(self,
                            polarization: str = 'circ',
                            normal_angle: float = .0,
                            gamma: float = None) -> pd.Series:

        gamma = self.__process_gamma_input(gamma)
        s_pol = self._polarization_func('s', normal_angle)
        p_pol = self._polarization_func('p', normal_angle)

        y = np.pi * (gamma + self.__g) / np.sin(np.pi * gamma)
        u_s = np.sqrt((1 - np.power(s_pol / y, 2)) / (1 + np.power(self.__f * s_pol / y, 2)))
        u_p = np.sqrt((1 - np.power(p_pol / y, 2)) / (1 + np.power(self.__f * p_pol / y, 2)))

        if polarization.lower() == 's':
            r_s_max = (1 - u_s) / (1 + u_s)
            r_s_max.name = 's-polarization'
            return r_s_max

        elif polarization.lower() == 'p':
            r_p_max = (1 - u_p) / (1 + u_p)
            r_p_max.name = 'p-polarization'
            return r_p_max

        elif polarization.lower() in ['c', 'circ', 'circular']:
            r_circ_max = 0.5 * ((1 - u_s) / (1 + u_s) + (1 - u_p) / (1 + u_p))
            r_circ_max.name = 'circular-polarization'
            return r_circ_max

        else:
            raise ValueError('Wrong polarization. Must be "s", "p" or "circ".')

    # FIXME
    # def calc_resolving_power(self,
    #                          pol: str = 's',
    #                          normal_angle: float = .0,
    #                          gamma: float = None) -> pd.Series:
    #
    #     gamma = self.__process_gamma_input(gamma)
    #     ang_rad = normal_angle / 180 * np.pi
    #     pol = self._polarization_func(pol, normal_angle)
    #
    #     y = self.__calc_y(gamma)
    #
    #     c = 2 * np.sqrt(2) / 3 * np.pi
    #     im_mu = gamma * self.permittivity.e2_a + (1 - gamma) * self.permittivity.e2_s
    #
    #     res_power = pd.Series(np.zeros(im_mu.shape[0]), index=im_mu.index, dtype=np.float64)
    #     for i, en in enumerate(self.permittivity.index.values):
    #         xi = abs(self.__f.iloc[i] * pol) / y.iloc[i]
    #         if xi >= 10:
    #             res_power.iloc[i] = c / np.sin(np.pi * gamma.iloc[i]) * np.power(np.cos(ang_rad), 2) / \
    #                                 abs(pol) / (self.permittivity.loc[en, 'e1_a'] - self.permittivity.loc[en, 'e1_s'])
    #
    #         else:
    #             res_power.iloc[i] = np.power(np.cos(ang_rad), 2) / im_mu.iloc[i] / \
    #                                 np.power((1 - np.power(pol / y.iloc[i], 2)) *
    #                                          (1 + np.power(self.__f.iloc[i] * pol / y.iloc[i], 2)), 0.75)
    #     return res_power

    def calc_penetration_depth(self,
                               normal_angle: float = .0,
                               gamma: float = None) -> pd.Series:

        e = self.permittivity
        gamma = self.__process_gamma_input(gamma)
        s_pol = self._polarization_func('s', normal_angle)
        y = self.__calc_y(gamma)

        wavelengths = pd.Series(HC_CONST / e.index, index=e.index)
        im_mu = gamma * e.e2_a + (1 - gamma) * e.e2_s

        return wavelengths * np.cos(normal_angle) / np.pi / im_mu / np.sqrt(
            (1 - np.power(s_pol / y, 2)) * (1 + np.power(self.__f * s_pol / y, 2)))

    def calc_efficient_number_of_periods(self,
                                         normal_angle: float = .0,
                                         gamma: float = None):
        return self.calc_penetration_depth(normal_angle, gamma) / self.calc_period_thickness(normal_angle, gamma)

    def calc_period_thickness(self,
                              normal_angle: float = .0,
                              gamma: float = None) -> pd.Series:
        ang_rad = normal_angle / 180 * np.pi
        gamma = self.__process_gamma_input(gamma)

        delta_merged = self.absorber.opt_consts.join(self.spacer.opt_consts, how='outer',
                                                     lsuffix='_a', rsuffix='_s').interpolate('index')
        delta_eff = delta_merged.delta_a * gamma + (1 - gamma) * delta_merged.delta_s

        return HC_CONST / self.permittivity.index / 2 / np.cos(ang_rad) * \
            np.sqrt(1 - delta_eff / np.power(np.cos(ang_rad), 2))

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

    # def __getitem__(self, energy) -> pd.DataFrame:
    #     return self.permittivity.loc[energy, :]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(absorber=Compound("{self.mirror.split("/")[0]}"), ' \
               f'spacer=Compound("{self.mirror.split("/")[1]}"))'

    def __str__(self) -> str:
        return f'{self.mirror}'


if __name__ == '__main__':
    mo_be = BiMirror(Compound('CrN'), Compound('Sc'))
    print(mo_be.calc_efficient_number_of_periods()[350:400])
