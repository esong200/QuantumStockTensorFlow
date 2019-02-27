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

    def get_sentiment(self, content):
        client = language_v1.LanguageServiceClient()
        if isinstance(content, six.binary_type):
            content = content.decode('utf-8')

        type_ = enums.Document.Type.PLAIN_TEXT
        document = {'type': type_, 'content': content}

        response = client.analyze_sentiment(document)
        sentiment = response.document_sentiment
        returnValue = []
        returnValue.append(sentiment.score)
        returnValue.append(sentiment.magnitude)
        # print(returnValue)
        return returnValue

if __name__ == '__main__':
    agent = Analyzer(sys.argv[1], sys.argv[2])
    print(agent.get_sentiment('This computer is amazing. Overall, the processor is fast and the gpu is very powerfull, however, the screen is kind of small'))

