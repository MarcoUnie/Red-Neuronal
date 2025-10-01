historial = []

def agregar_entrada(tipo, entrada, resultado, archivo=None):
    historial.append({
        "tipo": tipo,         
        "entrada": entrada,   
        "resultado": resultado,
        "archivo": archivo,  
    })

def obtener_historial():
    return historial
