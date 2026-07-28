# gemini_prompts.py
"""
This file contains prompts for the Gemini API.

Before running this code:

`pip install google-genai`

- Make sure to set the GEMINI_API_KEY environment variable with your API key.

Go to https://developers.generativeai.google/ to get your API key.
Copy the API key and set it in your environment variables. 
For example, in Linux or macOS, you can use:
`echo 'export GEMINI_API_KEY="YOUR_API_KEY"' >> ~/.zshrc`

AND do not forget to run:

`source ~/.zshrc`
"""

from google import genai

# The client retrieves automatically the GEMINI_API_KEY environment variable
client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.5-flash",  # Standard and fast model, or "gemini-3.5-pro" for complex tasks
    contents="Summarize NLP's main goal in one sentence.",
)

print(response.text)