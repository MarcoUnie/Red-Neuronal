from flask import Flask, request, render_template
from compiler import compile_model

app = Flask(__name__)

# Página principal
@app.route("/", methods=["GET", "POST"])
def index():
    modelo_resumen = None
    error = None

    if request.method == "POST":
        arquitectura = request.form.get("arquitectura")
        try:
            # Compilar el modelo
            modelo = compile_model(arquitectura, input_shape=(784,))
            
            # Capturar el resumen en texto
            from io import StringIO
            stream = StringIO()
            modelo.summary(print_fn=lambda x: stream.write(x + "\n"))
            modelo_resumen = stream.getvalue()

        except Exception as e:
            error = str(e)

    return render_template("web.html", modelo_resumen=modelo_resumen, error=error)
    
if __name__ == "__main__":
    app.run(debug=True)

#Respuesta al analisis 1:Al tener 10 millones de parámetros la expresiones se volverían largas y propensas a
#errores, además se necesitaría un codigo muy bien optimizado para no quedarse sin memoria. Por lo tanto 
#realmente el desafío se encuentra en hacerlo eficiente y estable.

#Respuesta al analisis 2:Este nivel de abstracción es mucho más claro que construirlo capa por capa, además permite
#que personas sin mucho conocimiento en la programación sean capaces de crear modelos con facilida, y por último
#se podría compilar también en otras librerías como pytorch o keras.