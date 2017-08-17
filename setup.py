from setuptools import setup

setup(name='dans_pymodules',
      version='3.8',
      description='Useful little modules that I likely need in more than one application',
      url='https://github.com/DanielWinklehner/dans_pymodules',
      author='Daniel Winklehner, Philip Weigel',
      author_email='winklehn@mit.edu',
      license='MIT',
      packages=['dans_pymodules'],
      package_data={'': ['PlotSettingsDialog.glade', 'header.tex', 'fishfinder.png', 'vitruvian.jpg']},
      include_package_data=True,
      zip_safe=False)
