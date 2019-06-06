from flask import Flask, request

app = Flask(__name__)

@app.route('/summarize', methods=['POST'])
def summarize():
    content = request.form.get("email_content")
    print("Receive Content: {}".format(content))
    summary = "Greetings from the server!"
    return summary
