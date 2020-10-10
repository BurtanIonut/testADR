from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def form_input():
    return render_template("textbox_input.html")


@app.route('/', methods=['POST'])
def form_input_post():
    text = request.form['text']
    print(text)
    return render_template("textbox_input.html", data={"text": text})


if __name__ == '__main__':
    app.run(debug=True)
