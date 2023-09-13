# import 구문
from flask import (
    Flask,
    render_template,
    # current_app,
    # g,
    request,
    url_for,
    redirect,
    flash,
)
from email_validator import validate_email, EmailNotValidError

app = Flask(__name__)
# add config key
app.config["SECRET_KEY"] = "9sfno39asf8nk32"


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
        username = request.form["username"]
        email = request.form["email"]
        description = request.form["description"]

        is_valid = True

        if not username:
            flash("사용자명은 필수입니다.")
            is_valid = False
        if not email:
            flash("이메일은 필수입니다.")
            is_valid = False

        try:
            validate_email(email)
        except EmailNotValidError:
            flash("메일 주소가 형식이 올바르지 않습니다.")
            is_valid = False

        if not description:
            flash("문의 내용은 필수입니다.")
            is_valid = False

        if not is_valid:
            return redirect(url_for("contact"))

        # 이메일 보내기

        # contact 로 리다이렉트 하기
        flash("문의해 주셔서 감사합니다.")
        return redirect(url_for("contact_complete"))

    return render_template("contact_complete.html")
