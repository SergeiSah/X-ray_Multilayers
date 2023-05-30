import os
import pandas as pd
import numpy as np
from typing import Union
import xraydb as xdb
from chemparse import parse_formula
from sqlalchemy import MetaData, Table, create_engine, select
from mp_api.client import MPRester


HC_CONST = 12398.41984  # planck_constant * light_speed [eV * A]


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
            self.name = element
            self.Z = None
        elif isinstance(element, int):
            self.Z = element
            self.name = None
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

    def __init__(self, chem_formula: str, is_download: bool = True):
        if xdb.validate_formula(chem_formula):
            self.chem_formula = chem_formula
        else:
            raise ValueError('Invalid chemical formula')

        self.chem_name = None
        self.molar_mass = None

        try:
            self.__get_properties()
        except ValueError as msg:
            print(f'ValueError: {msg}')

            if is_download:
                print('Downloading from the Materials Project...')
                self.__download_properties()

        self.__get_coefficients()

    @property
    def stoichiometry(self):
        return xdb.chemparse(self.chem_formula)

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
        cs = cs.interpolate()

        # calculate sum of factors of elements for compound
        cs['f1'] = cs.filter(like='f1').sum(axis=1)
        cs['f2'] = cs.filter(like='f2').sum(axis=1)

        p = 2.7008645E-6  # n * r_0 / 2 / pi, r_0 - classical electron radius, n - number density of atoms
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
    e = Compound('Au')
    e.add_energies([1000])
    print(e.factors.query('energy in (1000,)'))
    print(e.opt_consts.query('energy in (1000,)'))
    print(e.permittivity.query('energy in (1000,)'))