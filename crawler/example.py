import sqlite3
from bs4 import BeautifulSoup
import urllib.request

req = urllib.request.urlopen('http://royalbreath.com')
html = req.read()

gei = BeautifulSoup(html, "html.parser")

conn = sqlite3.connect('test_db.db')

c = conn.cursor()

c.execute('''CREATE TABLE stocks(date TEXT, trans TEXT, symbol TEXT, qty REAL, price REAL)''')
conn.commit()
conn.close()
print(gei.prettify())
