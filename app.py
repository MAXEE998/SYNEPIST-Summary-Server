from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/summarize', methods=['POST'])
def summarize():
    content = request.form.get("email_content")
    print("Receive Content: {}".format(content))
    summary = "Greetings from the server!"
    return jsonify(summary=summary)
