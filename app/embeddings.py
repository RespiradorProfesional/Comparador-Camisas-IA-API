import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import tensorflow as tf
import json
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageFilter
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Cargar el modelo de embeddings previamente entrenado
modelo_embeddings = load_model('model/Modelo_final_embenddings_ropa.keras')

# Función para procesar imágenes antes de convertirlas a embeddings
def procesar_imagen(image_path, resize_to=(299, 299), grayscale=False, apply_filter=None):
    """
    Procesa una imagen. La redimensiona, la convierte a escala de grises si es necesario,
    y aplica un filtro si se proporciona.
    
    :param image_path: Ruta de la imagen a procesar.
    :param resize_to: Dimensiones a las que redimensionar la imagen.
    :param grayscale: Indica si la imagen debe ser convertida a escala de grises.
    :param apply_filter: Filtro de imagen a aplicar, si se proporciona.
    :return: Imagen procesada.
    """
    try:
        with Image.open(image_path) as img:
            print(f"Procesando imagen desde: {image_path}")
            
            # Redimensionar la imagen
            img = img.resize(resize_to)
            
            # Convertir a escala de grises si se solicita
            if grayscale:
                img = img.convert("L")
            
            # Aplicar un filtro si se proporciona
            if apply_filter:
                img = img.filter(apply_filter)
            
            return img
    
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None

# Función para convertir una imagen a su embedding correspondiente
def image_to_embedding(image_user: Image.Image):
    """
    Convierte una imagen en su embedding utilizando el modelo previamente cargado.
    
    :param image_user: Imagen de entrada.
    :return: Embedding de la imagen.
    """
    # Redimensionar la imagen y convertirla a RGB si es necesario
    resized_image = image_user.resize((299, 299))
    if resized_image.mode not in ['RGB']:
        resized_image = resized_image.convert('RGB')

    # Convertir la imagen en un array de numpy y normalizarla
    image_array = image.img_to_array(resized_image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Obtener el embedding de la imagen usando el modelo cargado
    image_embedding = modelo_embeddings.predict(image_array)
    return image_embedding.flatten()

# Función para comparar dos embeddings usando la similitud del coseno
def compare_embeddings(embedding1, embedding2):
    """
    Compara dos embeddings utilizando la similitud del coseno.
    
    :param embedding1: Primer embedding.
    :param embedding2: Segundo embedding.
    :return: Similitud entre los dos embeddings.
    """
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Función para cargar los embeddings desde un archivo JSON
def get_embeddings_from_json(json_file_path):
    """
    Carga los embeddings de imágenes almacenados en un archivo JSON.
    
    :param json_file_path: Ruta del archivo JSON con los embeddings.
    :return: Lista de tuplas con el ID del producto y su embedding correspondiente.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    image_data = []
    for item in data:
        product_id = item.get("id")
        embedding = item.get("embedding")
        if product_id and embedding:
            image_data.append((product_id, embedding))

    return image_data

# Función principal para obtener la similitud entre una imagen de referencia y un conjunto de imágenes
def get_similarity_score(reference_image_url: str, image_array: list):
    """
    Calcula la similitud entre una imagen de referencia y un conjunto de imágenes.
    
    :param reference_image_url: URL o ruta de la imagen de referencia.
    :param image_array: Lista de rutas de imágenes a comparar.
    :return: Lista de imágenes ordenadas por similitud con la imagen de referencia.
    """
    # Cargar la imagen de referencia y obtener su embedding
    reference_image = procesar_imagen(reference_image_url)
    reference_embedding = image_to_embedding(reference_image)

    similarities = []
    
    # Comparar la imagen de referencia con cada imagen en el array
    for idx, img_path in enumerate(image_array):
        img = procesar_imagen(img_path)
        img_embedding = image_to_embedding(img)
        similarity_score = cosine_similarity([reference_embedding], [img_embedding])
        similarities.append((idx, similarity_score[0][0]))

    # Ordenar las imágenes por similitud de mayor a menor
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Mostrar las imágenes más similares
    print("Imágenes más similares a la imagen de referencia:")
    for idx, score in similarities:
        print(f"Imagen {idx + 1}, Similaridad: {score}")

    return similarities

# Función para mostrar imágenes utilizando Matplotlib
def show_image(image_paths):
    """
    Muestra las imágenes en la lista de rutas usando Matplotlib.
    
    :param image_paths: Lista de rutas de imágenes a mostrar.
    """
    print(image_paths)
    for path in image_paths:
        img = mpimg.imread(path)
        imgplot = plt.imshow(img)
        plt.show()
