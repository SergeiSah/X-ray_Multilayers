import pandas as pd
import numpy as np
import plotly.express as px

from compounds import Compound
from bi_mirrors import BiMirror


class OptConstPlotter:

    def __init__(self, *materials: str | tuple[str, float]):
        self.__downloaded_materials = []
        self.__densities = []

        self.__delta = pd.DataFrame()
        self.__beta  = pd.DataFrame()
        self.__abs_coeff = pd.DataFrame()

        self.add_materials(*materials)

        self.image = {
            'width': 1000,
            'height': 600,
            'template': 'seaborn',
            'font': {'size': 20, 'family': 'Work Sans'}
        }

        self.backgrkound = {
            'plot_bgcolor': '#EFF5F5',
        }

        self.grid = {
            'gridcolor': '#D6E4E5',
            'gridwidth': 1.5
        }

        self.borders = {
            'mirror': True,
            'showline': True,
            'linecolor': 'black',
            'linewidth': 2
        }

    @property
    def delta(self):
        return self.__delta

    @property
    def beta(self):
        return self.__beta

    @property
    def abs_coeff(self):
        return self.__abs_coeff

    def add_materials(self, *materials: str | tuple[str, float]) -> None:
        """
        Add optical and absorption coefficients of the materials to the plotter.
        :param materials: if only the name of a material is given, the density will be taken from the database,
                          otherwise one must specify the name and the density in a list
        :return: None
        """

        materials = [(material, None) if isinstance(material, str) else material for material in materials]

        for material, density in materials:
            if (material, density) not in self.__downloaded_materials:
                comp = Compound(material, density=density)

                dens = density if density is not None else comp.density
                self.__downloaded_materials.append((material, dens))

                delta = comp.opt_consts[['delta']]
                delta.columns = [material]

                beta = comp.opt_consts[['beta']]
                beta.columns = [material]

                abs_coeff = comp.abs_coeff.to_frame()
                abs_coeff.columns = [material]

                self.__delta = pd.concat([self.delta, delta], axis=1)
                self.__beta  = pd.concat([self.beta, beta], axis=1)
                self.__abs_coeff = pd.concat([self.abs_coeff, abs_coeff], axis=1)

        self.__delta = self.__delta.sort_index().interpolate(method='index').dropna()
        self.__beta = self.__beta.sort_index().interpolate(method='index').dropna()
        self.__abs_coeff = self.__abs_coeff.sort_index().interpolate(method='index').dropna()

    def add_energies(self, energies: list[float]) -> None:
        not_in_index = [en for en in energies if en not in self.__delta.index]

        if not_in_index:
            for const in [self.__delta, self.__beta, self.__abs_coeff]:

                for energy in energies:
                    const.loc[energy, :] = np.nan

                const.sort_index(inplace=True)
                const.interpolate(method='index', inplace=True)

    def plot_opt_consts(self, materials: list[str | tuple[str, float]],
                        energy: float, x: str = 'n') -> None:
        """
        materials: list
            list of materials
        energy: float
            energy in eV
        x: str (default: 'n')
            x axis "delta" or "n"
        """

        self.add_materials(*materials)
        self.add_energies([energy])
        materials = [material if isinstance(material, str) else material[0] for material in materials]

        beta_n = pd.concat([self.delta.loc[energy, materials].to_frame().T, self.beta.loc[energy, materials].to_frame().T])
        beta_n.index = ['delta', 'beta']
        beta_n = beta_n.T.reset_index().rename(columns={'index': 'material'})
        beta_n['n'] = 1 - beta_n['delta']

        fig = px.scatter(beta_n, x=x, y='beta', text='material')

        fig.update_traces(textposition='top center', marker=dict(size=15, color='Green',
                                                                 line=dict(width=2, color='DarkSlateGrey')))
        fig.add_annotation(text=f'Energy = {energy} eV', showarrow=False, bgcolor='orange', xref='paper', yref='paper',
                           x=1, y=1, font=dict(color='DarkSlateGrey', size=24), borderpad=10)

        fig.update_layout(**self.image, **self.backgrkound)
        fig.update_yaxes(**self.borders, **self.grid, title_text=r'Absorption index', zeroline=False)
        fig.update_xaxes(**self.borders, **self.grid, title_text='Re(n)' if x == 'n' else 'Refraction index decrement')

        fig.show()

    def plot_abs_coeff(self, materials: list[str | tuple[str, float]],
                       en_range: list[float], xdtick=20, log_y=True) -> None:

        self.add_materials(*materials)
        self.add_energies(en_range)
        materials = [material if isinstance(material, str) else material[0] for material in materials]

        abs_coeffs = self.abs_coeff.loc[en_range[0]:en_range[-1], materials] * 1e+8  # convert from A^-1 to cm^-1

        fig = px.line(abs_coeffs, range_x=en_range, log_y=log_y, color_discrete_sequence=px.colors.qualitative.Prism)

        fig.update_layout(**self.image, **self.backgrkound, legend_title_text='Material')
        fig.update_xaxes(**self.borders, **self.grid, title='Energy (eV)', dtick=xdtick, ticks='outside')
        fig.update_yaxes(**self.borders, **self.grid, title='Absorption coefficient (cm<sup>-1</sup>)',
                         exponentformat='power', ticks='outside', zeroline=False)

        fig.show()

    def plot_delta_beta_ratio(self, spacer: str, absorbers: list[str], energy: float) -> None:
        self.add_materials(*absorbers, spacer)
        self.add_energies([energy])

        delta_absorb = self.delta.loc[energy, absorbers]
        delta_spacer = self.delta.loc[energy, spacer]

        beta_absorb = self.beta.loc[energy, absorbers]
        beta_spacer = self.beta.loc[energy, spacer]

        ratios = (abs(delta_absorb - delta_spacer) / (beta_absorb - beta_spacer)).sort_values(ascending=False)

        fig = px.bar(ratios, color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(**self.image, **self.backgrkound, showlegend=False)
        fig.update_layout(width=800)

        fig.add_annotation(text=f'Spacer: {spacer}', showarrow=False, bgcolor='orange', xref='paper', yref='paper',
                           x=1, y=0.95, font=dict(color='DarkSlateGrey', size=24), borderpad=10)

        fig.update_traces(width=0.7)
        fig.update_yaxes(**self.grid, **self.borders, zeroline=False, ticks='outside',
                         title='(δ<sub>a</sub> - δ<sub>s</sub>) / (β<sub>a</sub> - β<sub>s</sub>)')
        fig.update_xaxes(**self.grid, **self.borders, title='Absorber')

        fig.show()


class BiMirrorsPlotter:

    def __init__(self) -> None:
        self.image = {
            'width': 1000,
            'height': 600,
            'template': 'seaborn',
            'font': {'size': 20, 'family': 'Work Sans'}
        }

        self.background = {
            'plot_bgcolor': '#EFF5F5',
        }

        self.grid = {
            'gridcolor': '#D6E4E5',
            'gridwidth': 1.5
        }

        self.borders = {
            'mirror': True,
            'showline': True,
            'linecolor': 'black',
            'linewidth': 2
        }

    def plot_param(self,
                   bi_mirrors: list[str],
                   periods: list[float],
                   gammas: list[float] | str = 'opt',
                   param: str = 'r_max',
                   legend: str = 'mirror',
                   en_range: tuple[float, float] | None = None,
                   x_scale: str = 'eV') -> None:

        mirror_param = pd.DataFrame()
        y_title = {
            'r_max': 'Peak reflectivity',
            'opt_gamma': 'Optimal γ=d<sub>a</sub>/d',
            'n_eff': 'Effective number of periods'
        }

        # calculate desired parameter
        for bi_mirror, period, gamma in zip(bi_mirrors, periods, gammas):
            absorber, spacer = bi_mirror.split('/')
            mirror = BiMirror(absorber, spacer)

            match param:
                case 'opt_gamma':
                    m_param = mirror.calc_optimal_gamma()
                case 'r_max':
                    angles = mirror.calc_normal_angle(period, gamma)
                    m_param = mirror.calc_max_reflection(normal_angle=angles, gamma=gamma)
                case 'n_eff':
                    m_param = mirror.calc_efficient_number_of_periods(gamma=gamma)
                case _:
                    raise ValueError(f'Wrong param. Must be "opt_gamma", "r_max" or "n_eff".')

            mirror_param = pd.concat([mirror_param, m_param], axis=1)

        match legend:
            case 'mirror':
                cols = bi_mirrors
            case 'period':
                cols = [f'{period / 10} nm' for period in periods]
            case 'gamma':
                cols = gammas
            case _:
                raise ValueError(f'Wrong legend. Must be "mirror", "period" or "gamma"')

        mirror_param.columns = cols
        mirror_param = mirror_param.sort_index().interpolate('index').dropna().loc[en_range[0]:en_range[1]]

        if x_scale == 'keV':
            mirror_param.index /= 1000

        fig = px.line(mirror_param, color_discrete_sequence=px.colors.qualitative.Prism)

        fig.update_layout(**self.image, **self.background, legend_title_text=legend.title())
        fig.update_xaxes(**self.borders, **self.grid, title=f'Energy ({x_scale})', ticks='outside')
        fig.update_yaxes(**self.borders, **self.grid, title=y_title[param], ticks='outside', zeroline=False)

        fig.show()

