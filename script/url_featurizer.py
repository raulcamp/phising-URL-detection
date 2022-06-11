import re
import math

class URLFeaturizer(object):
    def __init__(self, url):
        self.url = url

    ## URL string Features
    def entropy(self):
        string = self.url.strip()
        prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
        entropy = sum([(p * math.log(p) / math.log(2.0)) for p in prob])
        return entropy

    def numDigits(self):
        digits = [i for i in self.url if i.isdigit()]
        return len(digits)

    def numSpecial(self):
        regex = re.compile('[@_!#$%^&*()<>?|}{~]')
        return len([c for c in self.url if regex.search(c)])

    def urlLength(self):
        return len(self.url)

    def numParameters(self):
        params = self.url.split('&')
        return len(params) - 1

    def numFragments(self):
        fragments = self.url.split('#')
        return len(fragments) - 1

    def numSubDomains(self):
        subdomains = self.url.split('http')[-1].split('//')[-1].split('/')
        return len(subdomains)-1

    def hasHttp(self):
        return 1 if 'http:' in self.url else 0

    def hasHttps(self):
        return 1 if 'https:' in self.url else 0
    
    def hasPercent(self):
        return 1 if '%' in self.url else 0

    def extract(self):
        data = {}
        data['entropy'] = self.entropy()
        data['numDigits'] = self.numDigits()
        data['numSpecial'] = self.numSpecial()
        data['urlLength'] = self.urlLength()
        data['numParams'] = self.numParameters()
        data['numFragments'] = self.numFragments()
        data['numSubDomains'] = self.numSubDomains()
        data['hasHttp'] = self.hasHttp()
        data['hasHttps'] = self.hasHttps()
        data['hasPercent'] = self.hasPercent()
        return data

if __name__ == '__main__':
    URL = 'https://www.google.com'
    featurizer = URLFeaturizer(URL)
    features = featurizer.extract()
    print(features)