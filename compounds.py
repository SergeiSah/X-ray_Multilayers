import os
import pandas as pd
from typing import Union
from chemparse import parse_formula
from sqlalchemy import MetaData, Table, create_engine, select
from mp_api.client import MPRester


def connect_to_db(db_path: str, table_name: str):
    def actual_decorator(func):
        meta = MetaData()
        engine = create_engine(f'sqlite+pysqlite:///data_collections/{db_path}')
        data = Table(table_name, meta, autoload_with=engine)

        def wrapper(self):
            with engine.connect() as connection:
                func(self, connection, data)

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
            raise TypeError('parameter "element" must be name of an element or its atomic number')

        self.__get_properties()

    @connect_to_db('Elements.db', table_name='Element')
    def __get_properties(self, conn, elements):
        if self.name is not None:
            condition = elements.c.Element == self.name
        else:
            condition = elements.c.Z == self.Z

        query = select(elements.c.Z,
                       elements.c.Element,
                       elements.c.Atomic_weight,
                       elements.c.Density).select_from(elements).where(condition)

        props = conn.execute(query).fetchone()

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
        self.chem_formula = chem_formula
        self.chem_name = None
        self.molar_mass = None

        try:
            self.__get_properties()
        except ValueError as msg:
            print(f'ValueError: {msg}')

            if is_download:
                print('Downloading from the Materials Project...')
                self.__download_properties()

    @property
    def stoichiometry(self):
        return parse_formula(self.chem_formula)

    @connect_to_db('Compounds.db', table_name='Properties')
    def __get_properties(self, connection, compounds):
        query = select('*').select_from(compounds)
        all_props = pd.DataFrame(connection.execute(query).fetchall(),
                                 columns=['formula', 'chem_name', 'density', 'mol_weight', 'source_name', 'id'])

        props = all_props[all_props['formula'].apply(parse_formula) == self.stoichiometry].reset_index(drop=True)

        if props.empty:
            raise ValueError('no such compound in the database')

        self.molar_mass = props.mol_weight.values[0]
        self.chem_name = props.chem_name.values[0]
        self.densities = props[['density', 'source_name', 'id']]

    def __download_properties(self):
        with MPRester(self.MAPI_KEY) as mpr:
            results = min(mpr.summary.search(formula=[self.chem_formula]), key=lambda x: x.formation_energy_per_atom)

        self.molar_mass = self.__calculate_molar_mass()
        self.densities = pd.DataFrame({'density': [results.density],
                                       'source_name': ['Materials Project'],
                                       'id': [results.material_id]})

    def __calculate_molar_mass(self):
        return sum(Element(elem).at_weight * st for elem, st in self.stoichiometry.items())

    def __repr__(self):
        return f'Compound({self.chem_formula})'

    def __str__(self):
        return self.chem_formula


if __name__ == '__main__':
    pd.options.display.max_columns = None
    e = Compound('WBe12')
    print(e.densities)
    print(e.molar_mass)
    # c = Element('sd')
