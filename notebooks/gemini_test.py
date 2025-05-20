import sys
print("Python executable:", sys.executable)
print("Python sys.path:", sys.path)

import google.generativeai as genai
print("google.generativeai location:", genai.__file__)

genai.configure(api_key="AIzaSyD3fwu0lxa0CtmW7m-PC5v1JtH3NmEK7qw")
model = genai.GenerativeModel("gemini-pro")  # NOT "models/gemini-pro"
response = model.generate_content("Say hello in the style of a wildfire response AI.")
print(response.text)

