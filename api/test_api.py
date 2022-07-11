import requests

#Define API and Input
url = 'http://127.0.0.1:8000/predict/image'
file_path = '../images/MINI Cooper Roadster Convertible 2012.jpg'

#Prepare Input
files = {'file': (file_path, open(file_path, 'rb'), "image/jpeg")}
response = requests.post(url, files=files)

print('status code : ',response.status_code)
print(response.json())
