import os
from definitions import PACKAGES_DIR
import pandas as pd
import numpy as np
from typing import Union
import xraydb as xdb
from chemparse import parse_formula
from sqlalchemy import MetaData, Table, create_engine, select
from scipy.constants import pi, N_A, speed_of_light, physical_constants
from mp_api.client import MPRester
from warnings import warn
import plotly.express as px


R_e = physical_constants['classical electron radius'][0]    # [m]
PLANCK = physical_constants['Planck constant in eV/Hz'][0]
HC_CONST = PLANCK * speed_of_light * 1e+10                  # [eV * Å]


def connect_to_db(db_path: str, table_name: str):
    def actual_decorator(func):
        meta = MetaData()
        engine = create_engine(f'sqlite+pysqlite:///{PACKAGES_DIR}/data_collections/{db_path}')
        data = Table(table_name, meta, autoload_with=engine)

        def wrapper(*args, **kwargs):
            with engine.connect() as connection:
                kwargs['connection'] = connection
                kwargs['data'] = data
                return func(*args, **kwargs)

        engine.dispose()
        return wrapper
    return actual_decorator


class Element:

    def __init__(self, element: Union[str, int]):
        if isinstance(element, str):
            self.name, self.Z = element, None
        elif isinstance(element, int):
            self.name, self.Z = None, element
        else:
            raise TypeError('parameter "element" must be a name of an element or its atomic number')

        self.__get_properties()

    @connect_to_db('Elements.db', table_name='Element')
    def __get_properties(self, connection, data):
        if self.name is not None:
            condition = data.c.Element == self.name
        else:
            condition = data.c.Z == self.Z

        query = select(data.c.Z,
                       data.c.Element,
                       data.c.Atomic_weight,
                       data.c.Density).select_from(data).where(condition)

        props = connection.execute(query).fetchone()

        if props is None:
            raise ValueError('no such element in the database')

        if self.name is None:
            self.name = props[1]
        else:
            self.Z = props[0]

        self.at_weight = float(props[2])
        self.density = float(props[3])

    def __repr__(self):
        return f'Element({self.name})'

    def __str__(self):
        return self.name


class Compound:
    MAPI_KEY = os.environ.get('MAPI_KEY')

    @classmethod
    def set_mapi_key(cls, key):
        cls.MAPI_KEY = key

    def __init__(self, chem_formula: str, density: float = None, download_from_mp: bool = False):
        if xdb.validate_formula(chem_formula):
            self.chem_formula = chem_formula
        else:
            raise ValueError('Invalid chemical formula')

        self.chem_name = None
        self.molar_mass = None
        self.__density = density
        self.__added_energies = []  # trace all added energies

        try:
            self.__get_properties()
        except ValueError as msg:
            print(f'ValueError: {msg}')

            if download_from_mp:
                print('Downloading from the Materials Project...')
                self.__download_properties()

        # self.__get_coefficients()

    @property
    def stoichiometry(self):
        return parse_formula(self.chem_formula)

    @property
    def density(self):
        return self.__density

    @density.setter
    def density(self, val):
        self.__density = val

        # recalculate all coefficients and add all energies added earlier
        self.__get_coefficients()
        self.add_energies(self.__added_energies)

    @connect_to_db('Atomic_factors.db', table_name='Atomic_factors')
    def __get_coefficients(self, connection, data):
        cs = pd.DataFrame()  # dataframe for coefficients

        # get atomic scattering factors for each element and multiply it by stoichiometry
        for element, st in self.stoichiometry.items():

            query = select(data.c.energy,
                           data.c.f1,
                           data.c.f2)\
                .select_from(data)\
                .where(data.c.element == element)

            cs = cs.join(pd.read_sql(query, connection, index_col='energy')
                         .rename(columns={'f1': f'f1_{element}', 'f2': f'f2_{element}'}) * st, how='outer')

        # interpolate missing values
        cs = cs.interpolate('index')

        # calculate sum of factors of elements for compound
        cs['f1'] = cs.filter(like='f1').sum(axis=1)
        cs['f2'] = cs.filter(like='f2').sum(axis=1)

        p = N_A * R_e * 1e+10 / 2 / pi * 1e-24
        wavelength = HC_CONST / cs.index
        
        cs['delta'] = p * self.density / self.molar_mass * np.power(wavelength, 2) * cs.f1
        cs['beta'] = p * self.density / self.molar_mass * np.power(wavelength, 2) * cs.f2

        n = 1 - cs['delta']
        cs['e1'] = np.power(n, 2) - np.power(cs['beta'], 2)
        cs['e2'] = 2 * n * cs['beta']

        self.factors = cs[['f1', 'f2']]
        self.opt_consts = cs[['delta', 'beta']]
        self.permittivity = cs[['e1', 'e2']]
        self.abs_coeff = 4 * np.pi * cs['beta'] / wavelength

    @connect_to_db('Compounds.db', table_name='Properties')
    def __get_properties(self, connection, data):
        query = select('*').select_from(data)
        all_props = pd.DataFrame(connection.execute(query).fetchall(),
                                 columns=['formula', 'chem_name', 'density', 'mol_weight', 'source_name', 'id'])

        props = all_props[all_props['formula'].apply(parse_formula) == self.stoichiometry].reset_index(drop=True)

        if props.empty:
            raise ValueError(f'no such compound {self.chem_formula} in the database')

        self.molar_mass = props.mol_weight.values[0]
        self.chem_name = props.chem_name.values[0]
        
        if all(props.density.isna()):
            warn('no density value for the compound in the database')
        else:
            self.densities = props[['density', 'source_name', 'id']]

            if self.density is None:
                self.density = self.densities[~self.densities.density.isna()].density.values[0]

    def __download_properties(self):
        with MPRester(self.MAPI_KEY) as mpr:
            results = min(mpr.summary.search(formula=[self.chem_formula]), key=lambda x: x.formation_energy_per_atom)

        self.molar_mass = self.__calculate_molar_mass()
        self.densities = pd.DataFrame({'density': [results.density],
                                       'source_name': ['Materials Project'],
                                       'id': [results.material_id]})
        self.density = self.densities.density.values[0]

    def __calculate_molar_mass(self):
        return sum(Element(elem).at_weight * st for elem, st in self.stoichiometry.items())

    def add_energies(self, energies: list):
        # exclude energies that are in the factors
        energies = [energy for energy in energies if energy not in self.factors.index]
        self.__added_energies.extend(energies)

        if energies:
            for cs in [self.factors, self.opt_consts, self.permittivity]:
                for energy in energies:
                    cs.loc[energy, :] = np.nan

                cs.sort_index(inplace=True)
                cs.interpolate(method='index', inplace=True)

    def __repr__(self):
        return f'Compound(chem_formula={self.chem_formula}, density={self.density})'

    def __str__(self):
        return self.chem_formula


class OptConstPlotter:

    def __init__(self, *materials: str):
        self.__downloaded_materials = []
        
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

    def add_materials(self, *materials: str):
        for material in materials:
            if material not in self.__downloaded_materials:
                self.__downloaded_materials.append(material)
                comp = Compound(material)

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

    def add_energies(self, energies: list):
        not_in_index = [en for en in energies if en not in self.__delta.index]

        if not_in_index:
            for const in [self.__delta, self.__beta, self.__abs_coeff]:

                for energy in energies:
                    const.loc[energy, :] = np.nan

                const.sort_index(inplace=True)
                const.interpolate(method='index', inplace=True)

    def plot_opt_consts(self, materials: list[str], energy: float, x: str = 'n') -> None:
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
        
    def plot_abs_coeff(self, materials: list[str], en_range: list[float], xdtick=20, log_y=True) -> None:
        
        self.add_materials(*materials)
        self.add_energies(en_range)
        
        abs_coeffs = self.abs_coeff.loc[en_range[0]:en_range[-1], materials] * 1e+8  # convert from A^-1 to cm^-1
        
        fig = px.line(abs_coeffs, range_x=en_range, log_y=log_y, color_discrete_sequence=px.colors.qualitative.Prism)
        
        fig.update_layout(**self.image, **self.backgrkound, legend_title_text='Material')
        fig.update_xaxes(**self.borders, **self.grid, title='Energy (eV)', dtick=xdtick, ticks='outside')
        fig.update_yaxes(**self.borders, **self.grid, title='Absorbtion coefficient (A<sup>-1</sup>)',
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
                         title='|δ<sub>a</sub> - δ<sub>s</sub>| / (β<sub>a</sub> - β<sub>s</sub>)')
        fig.update_xaxes(**self.grid, **self.borders, title='Absorber')

        fig.show()

