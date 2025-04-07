import requests

def call_remote_function():
    payload = {"key": "value"}  # Customize your payload as needed
    # Replace 'server_ip_address' with your server's actual IP, e.g., '165.132.40.52'
    url = "http://165.132.40.52:5000/trigger"
  
    try:
        print("Sending request to the server...")
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()  # Raise an error for non-200 responses
        print("Server responded with:", response.json())
    except requests.exceptions.RequestException as e:
        print("Error calling remote function:", e)

if __name__ == '__main__':
    # Example condition check; adjust as necessary
    condition_met = True
    if condition_met:
        call_remote_function()
