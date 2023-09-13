from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def index():
    return "hellow, flaskbook!"


@app.route("/hello/string:<name>", methods=["GET", "POST"], endpoint="hello-endpoint")
def hello(name):
    return f"<h1>hello, {name}</h1>"


@app.route("/name/<name>")
def show_name(name):
    return render_template("index.html", name=name)
