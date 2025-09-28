historial = []

def agregar_entrada(tipo, entrada, resultado, archivo=None, grafico=None):
    historial.append({
        "tipo": tipo,         # "texto" o "imagen"
        "entrada": entrada,   # cadena o nombre de archivo
        "resultado": resultado,
        "archivo": archivo,   # ruta al archivo .txt de descarga
        "grafico": grafico    # gráfico Plotly como HTML
    })

def obtener_historial():
    return historial
