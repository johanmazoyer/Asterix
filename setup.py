from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='Asterix',
    version='2.3',
    description='Asterix: A simulation tool for high-contrast sensing and control algorithms',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/johanmazoyer/Asterix',
    author='Asterix Developers',
    author_email='johan.mazoyer@obspm.fr',
    license='BSD',
    include_package_data=True,
    packages=find_packages(),
    zip_safe=False,
    classifiers=[
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.8, 3.9, 3.10',
        ],
    keywords='Exoplanets imaging high-contrast coronagraphy',
    install_requires=['astropy', 'configobj', 'jupyter', 'matplotlib', 'numpy', 'pytest', 'scikit-image', 'scipy']
    )
