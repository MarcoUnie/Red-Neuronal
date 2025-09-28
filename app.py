import os
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from PIL import Image
import io, base64
import plotly.graph_objects as go
from historial import agregar_entrada, obtener_historial
from tensorflow.keras.datasets import mnist

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "resultados"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULTS_FOLDER"] = RESULTS_FOLDER

modelo = tf.keras.models.load_model("model.h5")

def generar_grafico(predicciones):
    fig = go.Figure(data=[
        go.Bar(x=list(range(10)), y=predicciones[0])
    ])
    fig.update_layout(
        title="Probabilidades por dígito",
        xaxis_title="Dígito",
        yaxis_title="Probabilidad"
    )
    return fig.to_html(full_html=False)

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    archivo_descarga = None
    grafico = None

    if request.method == "POST":
        if "texto" in request.form and request.form["texto"].strip() != "":
            texto = request.form["texto"]
            resultado = f"Texto recibido con {len(texto.split())} palabras."
            filename = "resultado_texto.txt"
            filepath = os.path.join(app.config["RESULTS_FOLDER"], filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(resultado)

            agregar_entrada("texto", texto, resultado, filename)

            archivo_descarga = filename
        elif "imagen" in request.files:
            imagen = request.files["imagen"]
            if imagen.filename != "":
                filename = secure_filename(imagen.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                imagen.save(filepath)

                img = tf.keras.preprocessing.image.load_img(filepath, color_mode="grayscale", target_size=(28,28))
                img_array = tf.keras.preprocessing.image.img_to_array(img).reshape(1, 784) / 255.0

                pred = modelo.predict(img_array, verbose=0)
                digito = np.argmax(pred)
                resultado = f"Modelo reconoce el dígito como: {digito}"

                result_file = f"resultado_{filename}.txt"
                result_path = os.path.join(app.config["RESULTS_FOLDER"], result_file)
                with open(result_path, "w", encoding="utf-8") as f:
                    f.write(resultado)

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

                grafico = f'<img src="data:image/png;base64,{b64}" alt="MNIST {digito}" />'

                agregar_entrada("imagen", filename, resultado, result_file, grafico)

                archivo_descarga = result_file

    return render_template("web.html", resultado=resultado, archivo_descarga=archivo_descarga, grafico=grafico)

@app.route("/descargar/<filename>")
def descargar(filename):
    filepath = os.path.join(app.config["RESULTS_FOLDER"], filename)
    return send_file(filepath, as_attachment=True)

@app.route("/historial")
def ver_historial():
    return render_template("historial.html", historial=obtener_historial())

if __name__ == "__main__":
    app.run(debug=True)

#Respuesta al analisis 1:Al tener 10 millones de parámetros la expresiones se volverían largas y propensas a
#errores, además se necesitaría un codigo muy bien optimizado para no quedarse sin memoria. Por lo tanto 
#realmente el desafío se encuentra en hacerlo eficiente y estable.

#Respuesta al analisis 2:Este nivel de abstracción es mucho más claro que construirlo capa por capa, además permite
#que personas sin mucho conocimiento en la programación sean capaces de crear modelos con facilida, y por último
#se podría compilar también en otras librerías como pytorch o keras.