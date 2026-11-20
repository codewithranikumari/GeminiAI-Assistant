
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_NAME = "gemini-3.8-flash"

client = genai.Client(api_key=API_KEY) if API_KEY else None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    if not client:
        return jsonify({
            "error": "Gemini API key missing. Check your .env file."
        }), 500

    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    try:
        # Keep only recent, valid user/model conversation turns.
        contents = []

        for item in history[-20:]:
            role = item.get("role")
            text = item.get("text", "")

            if role in ("user", "model") and isinstance(text, str) and text.strip():
                contents.append({
                    "role": role,
                    "parts": [{"text": text}]
                })

        contents.append({
            "role": "user",
            "parts": [{"text": message}]
        })

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents
        )

        answer = response.text

        if not answer:
            answer = "Gemini did not return a text response."

        return jsonify({"reply": answer})

    except Exception as e:
        app.logger.exception("Gemini request failed")
        return jsonify({
            "error": "Gemini request failed. Check your API key, model access, internet and quota.",
            "details": str(e)[:500]
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
