from flask import Flask, request, jsonify
from flask_cors import CORS
from summary import *

app = Flask(__name__)
CORS(app)

@app.before_first_request
def initialize():
    model_path = "/workspace/SYNEPIST-Summary-Server/assets/model.pt"
    vocab_path = "/workspace/SYNEPIST-Summary-Server/assets/src.Field"
    global summarizer
    summarizer = Summarizer(model_path, vocab_path)
    print("Summarizer Initiated")


@app.route('/summarize', methods=['POST'])
def get_summary():
    content = request.form.get("email_content")
    print("Receive Content: {}".format(content))
    summary = summarizer.summarize([content])
    return jsonify(summary=summary[0])

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=80)
