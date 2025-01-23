import os
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import numpy as np
from embeddings import get_embeddings_from_json, image_to_embedding, compare_embeddings, show_image
import json

# Inicialización de la aplicación FastAPI
app = FastAPI()

@app.post("/compare_image/")
async def compare_image(file: UploadFile = File(...)):
    """
    Endpoint para comparar una imagen cargada por el usuario con un conjunto de imágenes almacenadas
    en un archivo JSON, devolviendo los 5 productos más similares.

    :param file: Imagen cargada por el usuario en la solicitud.
    :return: Los IDs de los 5 productos más similares a la imagen cargada.
    """
    # Abrir la imagen recibida en la solicitud
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))
    
    # Obtener el embedding de la imagen cargada
    image_embedding = image_to_embedding(image)

    # Obtener las imágenes y embeddings almacenados en el archivo JSON
    image_data = get_embeddings_from_json('image_embeddings.json')
    
    # Lista para almacenar las similitudes
    similarities = []
    
    # Comparar la imagen recibida con las imágenes almacenadas
    for product_id, embedding in image_data:
        # Calcular la similitud entre los embeddings
        similarity = compare_embeddings(image_embedding, embedding)
        similarities.append((product_id, similarity))
    
    # Ordenar las imágenes por similitud de mayor a menor
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Obtener los 5 productos más similares
    top_5_similar = similarities[:5]

    # Extraer los IDs de los productos más similares
    top_5_similar_ids = [x[0] for x in top_5_similar]
    
    # Retornar los IDs de los 5 productos más similares
    return {"top_similar_products": top_5_similar_ids}
