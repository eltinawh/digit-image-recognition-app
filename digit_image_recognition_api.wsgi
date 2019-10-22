#! /usr/bin/python

import sys
sys.path.insert(0, "/var/www/digit_image_recognition_api")
sys.path.insert(0,'/opt/conda/lib/python3.6/site-packages')
sys.path.insert(0, "/opt/conda/bin/")

import os
os.environ['PYTHONPATH'] = '/opt/conda/bin/python'

from digit_image_recognition_api import app as application