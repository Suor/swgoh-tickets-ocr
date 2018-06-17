import json
import requests
import lxml.html


with open('config.json') as f:
    config = json.load(f)


response = requests.get(config['guild_url'])
root = lxml.html.fromstring(response.text)
cells = root.cssselect('table tbody td:first-child')
names = [cell.attrib['data-sort-value'] for cell in cells]

with open('dict.txt', 'w') as f:
    f.write(''.join(name + '\n' for name in names))
