from flask import Flask
from flask import request
import random
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import google
import google.oauth2.credentials
from google.auth import compute_engine
import google.auth.transport.requests
import requests
import os
import datetime
import logging


# Create an instance of the Flask class that is the WSGI application.
# The first argument is the name of the application module or package,
# typically __name__ when using a single module.
app = Flask(__name__)
app.logger.setLevel(logging.INFO)
auth_req = google.auth.transport.requests.Request()
creds = compute_engine.Credentials()

# VERTEX AI Endpoint
ENDPOINT_ID = os.getenv("ENDPOINT_ID", None)
if not(ENDPOINT_ID) :
    app.logger.fatal("ENDPOINT env variable not defined")

PROJECT_ID = os.getenv("PROJECT_ID", None)
if not(PROJECT_ID) :
    app.logger.fatal("PROJECT_ID env variable not defined")

REGION = os.getenv("REGION", None)
if not(REGION) :
    app.logger.fatal("REGION env variable not defined")

if not(ENDPOINT_ID) or not(PROJECT_ID) or not(REGION):
    URL=None
else:
    URL = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/europe-west4/endpoints/{ENDPOINT_ID}:predict"

TITLE = os.getenv("TITLE","")

headers={}

# Flask route decorators map / and /hello to the hello function.
# To add other resources, create functions that generate the page contents
# and add decorators to define the appropriate resource locators for them.

@app.route('/', methods=['GET', 'POST'])
def brain():
    global headers
    if not(URL):
        return "Invalid configuration. Check the logs", 500

    # renew token
    if not(creds.token) or creds.expired:
        app.logger.warning("refreshing access token")
        creds.refresh(auth_req)
        headers = {
            "Authorization": f"Bearer {creds.token}",
            "Content-Type" : "application/json"
        }
    app.logger.debug(headers)

    x = np.random.normal(size = (1024))

    req = { "instances" : [ x.tolist() ]}
    app.logger.debug(req)
    resp = requests.post(URL, headers = headers, json = req, allow_redirects=True)
    if resp.status_code != 200:
        app.logger.error(resp.status_code)
        try:
            app.logger.error(resp.json())
        except :
            app.logger.error(resp.content)

        return "Error returned by endpoint. Check the logs", 500

    js=resp.json()
    app.logger.info(f"""endpoint resp {resp.status_code} Model: {js["modelDisplayName"]} v{js["modelVersionId"]}""")
    img=js['predictions'][0]

    imgplot = plt.imshow(img,cmap='gray')
    img = io.BytesIO()
    plt.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)
    b64=base64.b64encode(img.getvalue())
    return f'''
    <!DOCTYPE html>
    <html lang="en"><head><title>{TITLE}</title>
    <meta charset="utf-8">
    <style>
    table, td, tr {{ 
    border: 1px solid;
    padding: 3px 5px 3px 10px;
    border-collapse: collapse;
    }}
    </style>
    </head><body>
    <table>
    <tr><td>timestamp<td>{datetime.datetime.now().isoformat()}
    <tr><td>nodel name<td>{js["modelDisplayName"]}
    <tr><td>model version<td>{js["modelVersionId"]}
    <tr><td colspan="2">
            <img alt="ai generated image" src="data:image/png;base64,{format(b64.decode('utf-8'))}">
            </table>
            <form><input type="submit" value="random"></form></body></html>
            '''

if __name__ == '__main__':
    # Run the app server on localhost:4449
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

