from setuptools import setup

setup(name='shearlexity',
      version = '0.0.1',
      install_requires = ['numpy','scipy','PyShearlets'],      
      dependency_links=['git+https://github.com/grlee77/PyShearlets.git#egg=PyShearlets'],
      py_modules=['shearlexity'],
      classifiers = [
          'Development Status :: 4 - Beta',
          "Intended Audience :: Science/Research",
          'License :: OSI Approved :: GNU General Public License (GPL)',
          'Operating System :: OS Independent :: Linux',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering',
        ],
      
)
# git+https://github.com/Turbo87/utm.git@v0.3.1#egg=utm-0.3.1
