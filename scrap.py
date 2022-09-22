# Import Modules
#coding: utf-8

import urllib2
import urllib
import cookielib
import re

# Google Url
google ='https://www.google.com/search?'

# Search Query
Query = "display motorola amoled moto x2 xt1097 com botões de volume e power"

# Set User Agent
header = [('User-Agent','Mozilla/5.0 (X11; Linux x86_64; rv:31.0) Gecko/20100101 Firefox/31.0 Iceweasel/31.8.0')]

# Create Cookie Handler
cj = cookielib.CookieJar()

# Create Url Handler 
url_opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj),  # Connect Cookie Jar
                                  urllib2.HTTPRedirectHandler())    # Address Redirect Handling Function

# Connect Header With Opener
url_opener.addheaders=header

# Encode Query With Url 
query = google + urllib.urlencode({'q':Query})

# Now Open Google Search Page
html = url_opener.open(query)

# Collect Html Code
codes = html.read()

# Compile Pattern
pattern = re.compile('<h3(.*?)</h3')

# list For Collecting results
collect_result = []

# Find Matches
for i in pattern.findall(codes):
    result = re.search('href=.(.*)..(onmousedown).+(>)([^><]+)(<)',i).groups()
    print result[0],result[3]
