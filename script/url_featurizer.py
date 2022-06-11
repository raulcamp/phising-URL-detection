import re
import math
import ipaddress

class URLFeaturizer(object):
    def __init__(self, url):
        self.url = url

    ## URL string Features
    def urlLength(self):
        return len(self.url)

    def numDigits(self):
        digits = [i for i in self.url if i.isdigit()]
        return len(digits)
    
    def numPeriods(self):
        return self.url.count('.') - 3

    def numSpecial(self):
        regex = re.compile('[@_!#$%^&*()<>=?|}{~-]')
        return len([c for c in self.url if regex.search(c)])

    def numParameters(self):
        params = self.url.split('&')
        return len(params) - 1

    def numFragments(self):
        fragments = self.url.split('#')
        return len(fragments) - 1

    def numSubDomains(self):
        subdomains = self.url.split('http')[-1].split('//')[-1].split('/')
        return len(subdomains)-1

    def hasHttps(self):
        return 1 if 'https:' in self.url else 0
    
    def hasPercent(self):
        return 1 if '%' in self.url else 0

    def hasAt(self):
        return 1 if '@' in self.url else 0

    def hasIP(self):
        try:
            ipaddress.ip_address(self.url)
            return 1
        except:
            return 0
    
    def hasPHP(self):
        return 1 if '.php' in self.url.lower() else 0

    def entropy(self):
        string = self.url.strip()
        prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
        entropy = sum([(p * math.log(p) / math.log(2.0)) for p in prob])
        return entropy

    def extract(self):
        data = {}
        data['urlLength'] = self.urlLength()
        data['numDigits'] = self.numDigits()
        data['numPeriods'] = self.numPeriods()
        data['numSpecial'] = self.numSpecial()
        data['numParams'] = self.numParameters()
        data['numFragments'] = self.numFragments()
        data['numSubDomains'] = self.numSubDomains()
        data['hasHttps'] = self.hasHttps()
        data['hasPercent'] = self.hasPercent()
        data['hasAt'] = self.hasAt()
        data['hasIP'] = self.hasIP()
        data['hasPHP'] = self.hasPHP()
        data['entropy'] = self.entropy()
        return data

if __name__ == '__main__':
    URL = 'https://www.google.com'
    featurizer = URLFeaturizer(URL)
    features = featurizer.extract()
    print(features)