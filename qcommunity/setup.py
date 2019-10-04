from setuptools import setup

setup(
    name='qcommunity',
    description=
    'Quantum Local Search framework for graph modularity optimization',
    author='Ruslan Shaydulin',
    author_email='rshaydu@g.clemson.edu',
    packages=['qcommunity'],
    install_requires=[
        'qiskit', 'qiskit_aqua', 'networkx', 'numpy', 'matplotlib', 'joblib', 'pyomo',
        'progressbar2', 'SALib', 'GPyOpt', 'sobol_seq'
    ],
    zip_safe=False)
