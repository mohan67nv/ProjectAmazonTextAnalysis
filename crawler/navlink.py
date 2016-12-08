import sqlite3
from bs4 import BeautifulSoup
import urllib.request


req = urllib.request.urlopen('https://www.vg.no')
html = req.read()

soup = BeautifulSoup(html, "html.parser")

mydivs = soup.findAll('div')
for div in mydivs:
    if (div["class"]=="a-spacing-top-small a-link-normal"):
        print (div)