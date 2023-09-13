# import 구문
from flask import (
    Flask,
    session,
    render_template,
    # current_app,
    # g,
    request,
    url_for,
    redirect,
    flash,
    make_response,
)
from email_validator import validate_email, EmailNotValidError
import logging
import os
from flask_debugtoolbar import DebugToolbarExtension
from flask_mail import Mail, Message


app = Flask(__name__)

# add config key
app.config["SECRET_KEY"] = "9sfno39asf8nk32"
app.config["DEBUG_TB_INTERCEPT_REDIRECTS"] = False
# mail config
app.config["MAIL_SERVER"] = os.environ.get("MAIL_SERVER")
app.config["MAIL_PORT"] = os.environ.get("MAIL_PORT")
app.config["MAIL_USE_TLS"] = os.environ.get("MAIL_USE_TLS")
app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = os.environ.get("MAIL_DEFAULT_SENDER")
toolbar = DebugToolbarExtension(app)

mail = Mail(app)
app.logger.setLevel(logging.DEBUG)

# # set key
# username = request.cookies.get("username")


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
    # response
    response = make_response(render_template("contact.html"))
    response.set_cookie("flaskbook key", "flaskbook value")
    # session
    session["username"] = "freshmea"

    return response


@app.route("/contact/complete", methods=["GET", "POST"])
def contact_complete():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        description = request.form["description"]

        try:
            if not username:
                flash("사용자명은 필수입니다.")
                raise Exception
            if not email:
                flash("이메일은 필수입니다.")
                raise Exception
            if not description:
                flash("문의 내용은 필수입니다.")
                raise Exception
            validate_email(email)
        except EmailNotValidError:
            flash("메일 주소가 형식이 올바르지 않습니다.")
            return redirect(url_for("contact"))
        except Exception:
            return redirect(url_for("contact"))
        else:
            # contact 로 리다이렉트 하기
            flash("문의해 주셔서 감사합니다.")
            # 이메일 보내기
            send_email(
                email,
                "문의 감사합니다.",
                "contact_mail",
                username=username,
                description=description,
            )
            return redirect(url_for("contact_complete"))

    return render_template("contact_complete.html")


def send_email(to, subject, template, **kwargs):
    msg = Message(subject, recipients=[to])
    msg.body = render_template(template + ".txt", **kwargs)
    msg.html = render_template(template + ".html", **kwargs)
    print("sending mail!!!!")
    mail.send(msg)
