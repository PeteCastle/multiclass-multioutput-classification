import requests

# Create a request to the localhost translation API
def translate(args):
    text, source = args
    url = "http://127.0.0.1:5000/translate"
    body = {
        "q": text,
        "source": source,
        "target": "en",
        "format": "text",
    }
    response = requests.post(url, json=body)

    if response.status_code != 200:
        raise Exception(response.json())

    return response.json()["translatedText"]