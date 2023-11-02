from flask import Flask, render_template, request
from Chat2 import cbfs  # Assuming your chatbot code is in a file named chat2.py

app = Flask(__name__, static_folder='/Users/aaliyahsalia/Desktop/SFBU/6thTrimester/CS589/Week6_HW3/static')

cb = cbfs()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    if not question:
        return "Please enter a question."
    widget_box = cb.convchain(question)
    answer = widget_box[1].object  # assuming the answer is the second widget in the box
    return render_template("index.html", question=question, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
