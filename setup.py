#!/usr/bin/python3

import os
import re

from setuptools import setup, find_packages

#Package meta-data.
NAME = 'pyzm'
DESCRIPTION = 'ZoneMinder API, Logger and other base utilities for python programmers'
URL = 'https://github.com/ZoneMinder/pyzmNg'
AUTHOR_EMAIL = 'info@zoneminder.com'
AUTHOR = 'Pliable Pixels'
LICENSE = 'GPL'
INSTALL_REQUIRES=[
    'requests>=2.18.4',
    'pydantic>=2.0.0',
    ]

# More time formats in events(since=...), such as "monday" or "January 5".
# Without it pyzm handles "N <unit> ago" (with the common abbreviations),
# "last week/month/year", now/today/yesterday and ISO timestamps.
_DATES_REQUIRES=[
    'dateparser>=1.1.0',
    ]

# The ZM database driver: DB logging, ZM log levels, event tagging, event
# paths and BirdNET audio. Not needed for API-only use. mysql-connector-python
# is still used when it is installed (installs from pyzm <= 2.5.4 have it).
# <2: the log handler calls ping(reconnect=False), and PyMySQL has deprecated
# that argument; a major release may drop it.
_DB_REQUIRES=[
    'PyMySQL>=1.0,<2',
    ]

_ML_REQUIRES=[
    'numpy>=1.13.3',
    'Pillow',
    'onnx>=1.12.0',
    'Shapely>=1.7.0',
    'portalocker>=2.3.0',
    'imutils',
    ]


here = os.path.abspath(os.path.dirname(__file__))
# read the contents of your README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    f.close()

def read(*parts):
    with open(os.path.join(here, *parts), encoding='utf-8') as fp:
        data = fp.read()
        fp.close()
        return data

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(name = NAME,
      python_requires='>=3.10',
      version = find_version('pyzm','__init__.py'),
      description = DESCRIPTION,
      long_description = long_description,
      long_description_content_type='text/markdown',
      author = AUTHOR,
      author_email = AUTHOR_EMAIL,
      url = URL,
      project_urls={
          'Documentation': 'https://pyzmng.readthedocs.io/en/latest/',
          'Source': 'https://github.com/ZoneMinder/pyzmNg',
          'Bug Tracker': 'https://github.com/ZoneMinder/pyzmNg/issues',
      },
      license = LICENSE,
      classifiers=[
          'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
      ],
      install_requires=INSTALL_REQUIRES,
      extras_require={
          'db': _DB_REQUIRES,
          'dates': _DATES_REQUIRES,
          'ml': _ML_REQUIRES + _DB_REQUIRES,
          'serve': _ML_REQUIRES + [
              'fastapi>=0.100',
              'uvicorn>=0.20',
              'python-multipart>=0.0.5',
              'PyJWT>=2.0',
              'PyYAML>=5.0',
          ],
          'train': _ML_REQUIRES + [
              'ultralytics>=8.3',
              'streamlit>=1.41',
              'streamlit-drawable-canvas>=0.9',
              'st-clickable-images>=0.0.3',
              'PyYAML>=5.0',
          ],
          'full': _ML_REQUIRES + _DB_REQUIRES + _DATES_REQUIRES + [
              'fastapi>=0.100',
              'uvicorn>=0.20',
              'python-multipart>=0.0.5',
              'PyJWT>=2.0',
              'PyYAML>=5.0',
              'ultralytics>=8.3',
              'streamlit>=1.41',
              'streamlit-drawable-canvas>=0.9',
              'st-clickable-images>=0.0.3',
          ],
      },
      package_data={'pyzm': ['train/assets/*.png']},
      packages=find_packages(exclude=["tests", "tests.*"]),
      )

