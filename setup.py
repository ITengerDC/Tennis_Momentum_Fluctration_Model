from setuptools import setup, find_packages

setup(
    name='Tennis_Momentum_Fluctration_Model',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'statsmodels',
        'matplotlib',
        'scipy',
        'xgboost',
        'scikit-learn',
        'seaborn',
    ],
)
