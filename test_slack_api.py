import requests

WEBHOOK_URL = "https://hooks.slack.com/services/AAA/BBB/CCC"
message = "recommendation"

response = requests.post(WEBHOOK_URL, json={"text": message})

if response.status_code == 200:
    print("✅ Message sent successfully")
else:
    print(f"⚠️ Failed to send message: {response.status_code}, {response.text}")

