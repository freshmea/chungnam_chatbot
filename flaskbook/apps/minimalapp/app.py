from flask import Flask, render_template, current_app, g, request, url_for, redirect

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


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/contact/complete", methods=["GET", "POST"])
def contact_complete():
    if request.method == "POST":
        # 이메일 보내기
        # contact 로 리다이렉트 하기
        return redirect(url_for("contact_complete"))

    return render_template("contact_complete.html")
