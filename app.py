import os
from flask import Flask, request, render_template, send_file, url_for
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from PIL import Image
import io, base64
from historial import agregar_entrada, obtener_historial
from tensorflow.keras.datasets import mnist

app = Flask(__name__,static_folder="resultados", static_url_path="/resultados")
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "resultados"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULTS_FOLDER"] = RESULTS_FOLDER

modelo = tf.keras.models.load_model("model.h5")

def cargar_imagen_como_base64(ruta):
    with open(ruta, "rb") as f:
        imagen_bytes = f.read()
    return "data:image/png;base64," + base64.b64encode(imagen_bytes).decode("utf-8")

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    archivo_descarga = None
    precision_final = None
    grafico_precision = None
    grafico_perdidas = None

    if request.method == "POST":
        imagen = request.files["imagen"]
        if imagen.filename != "":
            filename = secure_filename(imagen.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            imagen.save(filepath)

            img = tf.keras.preprocessing.image.load_img(filepath, color_mode="grayscale", target_size=(28,28))
            img_array = tf.keras.preprocessing.image.img_to_array(img).reshape(1, 784) / 255.0
            img_array = 1.0 - img_array

            pred = modelo.predict(img_array, verbose=0)
            digito = np.argmax(pred)
            resultado = f"Modelo reconoce el dígito como: {digito}"

            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            indices = np.where(y_test == digito)[0]
            if len(indices) == 0: 
                indices = np.where(y_train == digito)[0]
                source = x_train
            else:
                source = x_test

            idx = np.random.choice(indices)
            match_img = source[idx]

            pil_img = Image.fromarray(match_img.astype(np.uint8), mode='L')
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            pil_img = Image.fromarray(match_img.astype(np.uint8), mode='L')
            pil_img = pil_img.resize((310, 310), Image.NEAREST)
            img_filename = f"mnist_match_{digito}.png"
            img_path = os.path.join(app.config["RESULTS_FOLDER"], img_filename)
            pil_img.save(img_path)

            archivo_descarga = img_filename

            filepath = "resultados/precision_final.txt"
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    precision_final = f.read()

            agregar_entrada("imagen", filename, resultado,archivo_descarga)

            grafico_perdidas = "perdidas.png"
            grafico_precision = "precision.png"
    return render_template("web.html", resultado=resultado, archivo_descarga=archivo_descarga, precision_final = precision_final, grafico_perdidas = grafico_perdidas, grafico_precision = grafico_precision)


@app.route("/descargar/<filename>")
def descargar(filename):
    filepath = os.path.join(app.config["RESULTS_FOLDER"], filename)
    return send_file(filepath, as_attachment=True)

@app.route("/historial")
def ver_historial():
    return render_template("historial.html", historial=obtener_historial())

if __name__ == "__main__":
    app.run(debug=True)

#Respuesta al análisis 1:Al tener 10 millones de parámetros la expresiones se volverían largas y propensas a
#errores, además se necesitaría un codigo muy bien optimizado para no quedarse sin memoria. Por lo tanto 
#realmente el desafío se encuentra en hacerlo eficiente y estable.

#Respuesta al análisis 2:Este nivel de abstracción es mucho más claro que construirlo capa por capa, además permite
#que personas sin mucho conocimiento en la programación sean capaces de crear modelos con facilida, y por último
#se podría compilar también en otras librerías como pytorch o keras.
