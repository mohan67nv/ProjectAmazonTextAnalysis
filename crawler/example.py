import sqlite3
from bs4 import BeautifulSoup
import urllib.request
import sys

req = urllib.request.urlopen('http://royalbreath.com')
html = req.read()

gei= BeautifulSoup(html, "html.parser")

conn = sqlite3.connect('finnit.db')

c = conn.cursor()

c.execute('''CREATE TABLE stocks
             (date text, trans text, symbol text, qty real, price real)''')
conn.commit()
conn.close()
print (gei)
