# imports
import os
import requests
import json
from requests_toolbelt import MultipartDecoder

# Server params
SERVER = 'localhost'
PORT = 5000
URL = 'http://{}:{}/'.format(SERVER, PORT)

# Files to send
files = { 'image.jpg': open('figures/samples/nike.jpg', 'rb') }

# Send file/s
response = requests.post(URL,
    files = files)

print('[RESPONSE] Status: {}'.format(response.status_code))

# If receiving one file with Flask send_file
# with open('response_file.png', 'wb') as f:
#     f.write(response.content)

# If receiving multiple files with MultipartEncoder
decoder = MultipartDecoder(response.content, response.headers['Content-Type'])
for part in decoder.parts:
    # Get headers
    headers = part.headers
    form_data = part.headers.get(b'Content-Disposition').decode('utf-8')
    # Get values
    name = form_data.split('name="')[1].split('";')[0]
    filename = form_data.split('filename="')[1].split('"')[0]
    image_path = os.path.join('figures/responses', filename)
    print('[RESPONSE] Writing {}'.format(filename))
    # Save image
    with open(image_path, 'wb') as f: 
        f.write(part.content)
