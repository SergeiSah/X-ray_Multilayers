from definitions import PACKAGES_DIR
import pandas as pd
import numpy as np
import xraydb as xdb
from chemparse import parse_formula
from sqlalchemy import MetaData, Table, create_engine, select
from scipy.constants import pi, N_A, speed_of_light, physical_constants
from warnings import warn


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

    def __init__(self, element: str | int):
        match element:
            case str():
                self.name, self.Z = element, None
            case int():
                self.name, self.Z = None, element
            case _:
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
    def __init__(self, chem_formula: str, density: float = None):
        if xdb.validate_formula(chem_formula):
            self.chem_formula = chem_formula
        else:
            raise ValueError('Invalid chemical formula')

        self.chem_name = None
        self.__molar_mass = None
        self.__density = None
        
        self.densities = pd.DataFrame(columns=['density', 'source_name', 'id'])
        self.__added_energies = []  # trace all added energies
        
        self.__get_properties(density)

    @property
    def stoichiometry(self) -> dict[str, float]:
        return parse_formula(self.chem_formula)

    @property
    def density(self) -> float:
        return self.__density

    @density.setter
    def density(self, val: float) -> None:
        self.__density = val

        # recalculate all coefficients and add all energies added earlier
        self.__get_coefficients()
        self.add_energies(self.__added_energies)

    @connect_to_db('Atomic_factors.db', table_name='Atomic_factors')
    def __get_coefficients(self, connection, data) -> None:
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
        
        cs['delta'] = p * self.density / self.__molar_mass * np.power(wavelength, 2) * cs.f1
        cs['beta'] = p * self.density / self.__molar_mass * np.power(wavelength, 2) * cs.f2

        n = 1 - cs['delta']
        cs['e1'] = np.power(n, 2) - np.power(cs['beta'], 2)
        cs['e2'] = 2 * n * cs['beta']

        self.factors = cs[['f1', 'f2']]
        self.opt_consts = cs[['delta', 'beta']]
        self.permittivity = cs[['e1', 'e2']]
        self.abs_coeff = 4 * np.pi * cs['beta'] / wavelength

    @connect_to_db('Compounds.db', table_name='Properties')
    def __load_properties(self, connection, data) -> pd.DataFrame:
        # get data from the database
        query = select('*').select_from(data)
        all_props = pd.DataFrame(connection.execute(query).fetchall(),
                                 columns=['formula', 'chem_name', 'density', 'mol_weight', 'source_name', 'id'])

        props = all_props[all_props['formula'].apply(parse_formula) == self.stoichiometry].reset_index(drop=True)

        return props

    def __get_properties(self, density: float = None) -> None:
        props = self.__load_properties()

        if props.empty and density is None:
            raise ValueError(f'no such compound {self.chem_formula} in the database')

        if props.empty:
            self.__molar_mass = self.__calculate_molar_mass()
            self.densities = pd.DataFrame(columns=['density', 'source_name', 'id'])
        else:
            self.__molar_mass = props.mol_weight.values[0]
            self.chem_name = props.chem_name.values[0]
            self.densities = props[['density', 'source_name', 'id']]
        
        if all(props.density.isna()) and density is None:
            raise ValueError('no density value for the compound in the database')
        elif density is None:
            self.density = self.densities[~self.densities.density.isna()].density.values[0]
        else:
            self.density = density
            manual_density = pd.DataFrame([[self.density, 'manual', None]], columns=['density', 'source_name', 'id'])
            self.densities = pd.concat([manual_density, self.densities]).reset_index(drop=True)

    def __calculate_molar_mass(self) -> float:
        return sum(Element(elem).at_weight * st for elem, st in self.stoichiometry.items())

    def add_energies(self, energies: list) -> None:
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

