from setuptools import setup, find_packages

__version__ = exec(open('src/fedex_generator/version.py').read())

setup(
    name='fedex_generator',
    version='0.0.1',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    project_urls={
        'Git': 'https://github.com/analysis-bots/FEDEx_Generator',
    },
    install_requires=[
        'wheel',
        'pandas',
        'numpy',
        'python-dotenv',
        'singleton-decorator',
        'ipython',
        'scipy',
        'paretoset'
    ]
)
