import os

import glob
import json

import requests


# For every file in fonts folder
for png in glob.glob('../fonts/*.png'):
    file = open(png, 'r')
    HEADERS = {
        "Content-Type": "image/png",
        "Content-Length": str(os.path.getsize(png)),
        "Authorization": {
            "oauth_consumer_key": "Ysxxa5HtKx18l4LkMfRY",
        },
    }

    filename = png.split('/')[2]
    response = requests.put(url="/container/fonts/"+filename, headers=HEADERS,
            data=file.read())
    print(response)
