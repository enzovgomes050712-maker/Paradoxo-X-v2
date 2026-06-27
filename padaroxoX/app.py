
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from core.brain import ParadoxoBrain

import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)





print("Diretório atual:", os.getcwd())
brain = ParadoxoBrain()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():

    mensagem = request.form.get("mensagem", "")

    arquivos_recebidos = []

    if "arquivos" in request.files:
        arquivos = request.files.getlist("arquivos")

        for arquivo in arquivos:

            if arquivo.filename == "":
                continue

            nome = secure_filename(arquivo.filename)

            caminho = os.path.join(
                UPLOAD_FOLDER,
                nome
            )

            arquivo.save(caminho)

            arquivos_recebidos.append(nome)

    if arquivos_recebidos:
        mensagem += "\n\nArquivos enviados:\n"
        mensagem += "\n".join(arquivos_recebidos)

    resposta = brain.chat(mensagem)

    return jsonify({
        "resposta": resposta
    })

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )

