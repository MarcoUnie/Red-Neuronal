from flask import Flask, request, render_template, jsonify
import numpy as np
from capa import Capa
from mlp import MLP

# Inicializar la app Flask
app = Flask(__name__)

# Crear un modelo MLP global (ejemplo: 3 → 4 → 3 → 1)
red = MLP([
    Capa(num_entradas=3, num_neuronas=4, activacion="relu"),
    Capa(num_entradas=4, num_neuronas=3, activacion="relu"),
    Capa(num_entradas=3, num_neuronas=1, activacion="sigmoide")
])

@app.route("/", methods=["GET", "POST"])
def index():
    prediccion = None
    if request.method == "POST":
        # Obtener entradas desde el formulario
        entradas = request.form.get("entradas")  # Ej: "0.1,0.2,0.3"
        if entradas:
            try:
                vector = np.array([list(map(float, entradas.split(",")))])
                pred = red.predecir(vector)
                prediccion = pred.tolist()
            except Exception as e:
                prediccion = f"Error: {str(e)}"

    return render_template("web.html", prediccion=prediccion)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    entradas = np.array(data["entradas"])
    pred = red.predecir(entradas)
    return jsonify({"predicciones": pred.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
