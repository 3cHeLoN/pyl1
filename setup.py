#!/usr/bin/env python

from distutils.core import setup

setup(name='pyl1',
      version='0.9',
      description='Tools for sparse regression',
      author='Folkert Bleichrodt',
      author_email='3368283-3chelon@users.noreply.gitlab.com',
      packages=['pyl1'],
      install_requires=[
          'numpy', 'scipy'],
      zip_safe=False)
