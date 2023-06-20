import os
import pandas as pd
import numpy as np
from typing import Union
import xraydb as xdb
from chemparse import parse_formula
from sqlalchemy import MetaData, Table, create_engine, select
from scipy.constants import pi, N_A, speed_of_light, physical_constants
from mp_api.client import MPRester


r_e = physical_constants['classical electron radius'][0]
Planck = physical_constants['Planck constant in eV/Hz'][0]
HC_CONST = Planck * speed_of_light * 1e+10  # [eV * A]


def connect_to_db(db_path: str, table_name: str):
    def actual_decorator(func):
        meta = MetaData()
        engine = create_engine(f'sqlite+pysqlite:///data_collections/{db_path}')
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

    def __init__(self, chem_formula: str, download_from_mp: bool = True):
        if xdb.validate_formula(chem_formula):
            self.chem_formula = chem_formula
        else:
            raise ValueError('Invalid chemical formula')

        self.chem_name = None
        self.molar_mass = None
        self.__density = None
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
        return xdb.chemparse(self.chem_formula)

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

        p = N_A * r_e * 1e+10 / 2 / pi * 1e-24
        cs['delta'] = p * self.density / self.molar_mass * np.power(HC_CONST / cs.index, 2) * cs.f1
        cs['beta'] = p * self.density / self.molar_mass * np.power(HC_CONST / cs.index, 2) * cs.f2

        n = 1 - cs['delta']
        cs['e1'] = np.power(n, 2) - np.power(cs['beta'], 2)
        cs['e2'] = 2 * n * cs['beta']

        self.factors = cs[['f1', 'f2']]
        self.opt_consts = cs[['delta', 'beta']]
        self.permittivity = cs[['e1', 'e2']]

    @connect_to_db('Compounds.db', table_name='Properties')
    def __get_properties(self, connection, data):
        query = select('*').select_from(data)
        all_props = pd.DataFrame(connection.execute(query).fetchall(),
                                 columns=['formula', 'chem_name', 'density', 'mol_weight', 'source_name', 'id'])

        props = all_props[all_props['formula'].apply(parse_formula) == self.stoichiometry].reset_index(drop=True)

        if props.empty:
            raise ValueError('no such compound in the database')

        self.molar_mass = props.mol_weight.values[0]
        self.chem_name = props.chem_name.values[0]
        self.densities = props[['density', 'source_name', 'id']]
        self.density = self.densities.density.values[0]

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
        return f'Compound({self.chem_formula})'

    def __str__(self):
        return self.chem_formula


if __name__ == '__main__':
    pd.options.display.max_columns = None
    e = Compound('B4C')
    print(e.density)
    e.add_energies([1000])
    print(e.opt_consts.query('energy in (1000,)'))
    e.density = 2.9
    print(e.density)
    print(e.opt_consts.query('energy in (1000,)'))
