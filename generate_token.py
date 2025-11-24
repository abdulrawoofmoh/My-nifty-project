from kiteconnect import KiteConnect

# --- YOUR KEYS ARE FILLED IN BELOW ---
api_key = "jnte5vc1eukpl9ex"
api_secret = "szrlluz0owhj1lix9sqo99veblt6l02f"

# 1. Initialise Kite App
kite = KiteConnect(api_key=api_key)

# 2. Get the Login URL
print("Step 1: Login to this URL in your browser:")
print(kite.login_url())
print("-" * 50)

# 3. User Input
request_token = input("Step 2: Paste the 'request_token' from the URL bar here: ")

# 4. Generate Session
try:
    data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = data["access_token"]
    
    print("\nSUCCESS! ✅")
    print(f"Your Access Token: {access_token}")
    print("\nSave this token! You will use it to place orders today.")
    
except Exception as e:
    print("\nError generating token:", e)
