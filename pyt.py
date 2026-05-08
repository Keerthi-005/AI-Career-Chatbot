from flask import Flask, jsonify, render_template, request
import json
import os
from sentence_transformers import SentenceTransformer
from groq import Groq

# ================= FLASK =================

app = Flask(__name__, static_folder='static')

# ================= G A =================
# Paste your key here

client = Groq(
    api_key="YOUR_KEY"
)

# ================= GLOBAL VARIABLES =================

model = None
job_data = []
job_texts = []

STATIC_FOLDER = os.path.join(
    app.root_path,
    'static'
)

# ================= LOAD SENTENCE TRANSFORMER =================

def load_model():

    global model

    try:

        model = SentenceTransformer('all-MiniLM-L6-v2')

        print("✅ AI Model Loaded")

    except Exception as e:

        print(f"❌ Model Error: {e}")

        model = None


# ================= LOAD JOB DATA =================

def load_recommendation_jobs():

    global job_data, job_texts

    try:

        data_path = 'data/cleaned.json'

        if not os.path.exists(data_path):

            data_path = os.path.join(STATIC_FOLDER, 'cleaned.json')

        if not os.path.exists(data_path):

            print("⚠️ cleaned.json not found. Skipping.")

            return

        with open(data_path, 'r', encoding='utf-8') as f:

            data = json.load(f)

        job_data.clear()
        job_texts.clear()

        for entry in data:

            if not entry.get('Title1'):
                continue

            title = entry['Title1'].strip()

            job_texts.append(f"Job Title: {title}")

            job_data.append({'title': title})

        print(f"✅ Loaded {len(job_data)} jobs.")

    except Exception as e:

        print(f"❌ Error loading jobs: {e}")

        job_data = []
        job_texts = []


# ================= HOME ROUTES =================

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/bot')
def bot_page():
    return render_template('bot.html')


@app.route('/recommend')
def recommend_page():
    return render_template('recommend.html')


# ================= CAREER GUIDE APIs (unchanged) =================

@app.route('/api/categories')
def get_categories():

    try:

        file_path = os.path.join(STATIC_FOLDER, 'bot.json')

        with open(file_path, 'r', encoding='utf-8') as f:

            jobs_data = json.load(f)

        return jsonify(list(jobs_data.keys()))

    except Exception as e:

        return jsonify({"error": str(e)}), 500


@app.route('/api/jobs')
def get_jobs_by_category():

    category = request.args.get('category')

    try:

        file_path = os.path.join(STATIC_FOLDER, 'bot.json')

        with open(file_path, 'r', encoding='utf-8') as f:

            jobs_data = json.load(f)

        if category not in jobs_data:
            return jsonify([])

        return jsonify(jobs_data[category])

    except Exception as e:

        return jsonify({"error": str(e)}), 500


# ================= GROQ AI CHATBOT =================

@app.route('/api/chat-ai', methods=['POST'])
def chat_ai():

    try:

        data = request.json

        user_msg = data.get("message", "")

        if not user_msg:

            return jsonify({"reply": "Please enter a message."})

        # ================= GROQ CALL =================

        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "system",
                    "content": (
    "You are an AI Career Assistant helping students and freshers. "
    "Help with career guidance, job suggestions, AI, Data Science, "
    "Software, Cloud, Cybersecurity careers, skill roadmaps, salary "
    "expectations in India and abroad, resume and interview tips. "
    "Give SHORT and CONCISE answers only. "
    "Maximum 3-4 bullet points. No long paragraphs. "
    "Be direct and to the point."
)
                },
                {
                    "role": "user",
                    "content": user_msg
                }
            ],

            max_tokens=200,

            temperature=0.7
        )

        reply = response.choices[0].message.content

        return jsonify({"reply": reply})

    except Exception as e:

        print("Groq Error:", e)

        return jsonify({
            "reply": f"⚠️ AI error: {str(e)}"
        })


# ================= HEALTH CHECK =================

@app.route('/api/health')
def health_check():

    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "jobs_loaded": len(job_data),
        "bot_data": os.path.exists(os.path.join(STATIC_FOLDER, 'bot.json'))
    })


# ================= ERROR HANDLERS =================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# ================= MAIN =================

if __name__ == '__main__':

    print("🚀 Starting AI Career Assistant...")

    load_model()

    load_recommendation_jobs()

    bot_path = os.path.join(STATIC_FOLDER, 'bot.json')

    if os.path.exists(bot_path):
        print("✅ Career bot data found.")
    else:
        print("⚠️ bot.json not found in static folder.")

    print("✅ System Ready")

    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )