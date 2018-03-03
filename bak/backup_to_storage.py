import os

import glob
import json

import requests


# For every file in fonts folder
for png in glob.glob('../fonts/*.png'):
    file = open(png, 'rb')
    HEADERS = {
        "Content-Type": "image/png",
        "Content-Length": str(os.path.getsize(png)),
        "oauth_consumer_key": "Ysxxa5HtKx18l4LkMfRY",
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_version": "1.0",
    }

    filename = png.split('/')[2]
    payload = dict(body=file.read())
    response = requests.put(url="http://restapi.fs.ncloud.com/datasets/fonts/"+filename,
            headers=HEADERS, data=payload)
    print(response)
