from google.cloud import language_v1
from google.cloud.language_v1 import enums
import sys
import os
import six
import json


class Analyzer:
    def __init__(self, infile, outfile):
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        googleAPI = "EthanGoogleAPIKey.json"
        googleAPIDirectory = os.path.join(THIS_FOLDER, googleAPI)
        self.infile = infile
        self.outfile = outfile
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = googleAPIDirectory
