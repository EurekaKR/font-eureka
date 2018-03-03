import json

import requests

HEADER = {
    "Content-Type": "image/png",
    # TODO: filesize
    "Content-Length": "filesize",
    "Authorization": {
        "oauth_consumer_key": "Ysxxa5HtKx18l4LkMfRY",
    },
}
# TODO: for every file in my server
file = open("asdf.png")
# TODO: add url, this contains storage path data
response = requests.put(url="", header=HEADER, data=json.load(file))
