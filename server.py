from flask import Flask, request, render_template, url_for
from test import spam_detect

app = Flask(__name__)

@app.route("/main")
def home():
    return render_template("layout.html")

@app.route("/result",methods=["POST"])
def output():
    form_data = request.form
    status = spam_detect(form_data["email"])
    return render_template("response.html",status=status)

if __name__ == "__main__":
    app.run(debug=True)

#python srever.py
#http://localhost:5000/main
