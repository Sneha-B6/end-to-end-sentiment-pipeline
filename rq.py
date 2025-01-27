import requests

url = "http://127.0.0.1:5000/predict"
params = {"review_text": "The movie was fantastic and exciting."}
response = requests.get(url, params=params)
print(response.json())
