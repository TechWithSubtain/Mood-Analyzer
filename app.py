from flask import Flask, request, jsonify, render_template
from textblob import TextBlob
from google import genai
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Flask app
app = Flask(__name__)

# Initialize Gemini client
client = genai.Client(api_key=api_key)

# List of angry/frustrated keywords
angry_keywords = [
    "bastard", "idiot", "stupid", "terrible", "hate", "useless",
    "damn", "annoying", "ridiculous", "sucks"
]

# Home page route
@app.route('/')
def home():
    return render_template("index.html")

# Analyze route
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    customer_message = data.get("message", "").strip()

    if not customer_message:
        return jsonify({"error": "Empty message"}), 400

    # Mood detection
    lower_message = customer_message.lower()
    if any(word in lower_message for word in angry_keywords):
        mood = "Angry / Frustrated"
    else:
        polarity = TextBlob(customer_message).sentiment.polarity
        if polarity > 0.2:
            mood = "Happy"
        elif polarity < -0.2:
            mood = "Frustrated"
        else:
            mood = "Neutral / Confused"

    # Prepare prompt for Gemini
    prompt = f"""
    Customer Message: "{customer_message}"
    Mood: {mood}
    Write a professional, polite, and helpful response matching the mood.
    """

    # Call Gemini API
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        ai_reply = response.text
    except Exception as e:
        ai_reply = f"Error generating response: {e}"

    return jsonify({
        "mood": mood,
        "reply": ai_reply
    })

# Optional test route
@app.route('/test')
def test_gemini():
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Hello world!"
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)