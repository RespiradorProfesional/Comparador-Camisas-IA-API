## API de Comparación de Imágenes basada en DeepFashion

Esta es una API desarrollada con FastAPI que permite comparar imágenes subidas por el usuario con un conjunto de imágenes predefinidas almacenadas en un archivo JSON. La API utiliza un modelo de aprendizaje automático basado en InceptionV3, finetuneado con el dataset de DeepFashion, para extraer embeddings de imágenes y comparar similitudes entre ellas.

# Requisitos
Para ejecutar esta API, debes tener un entorno virtual configurado y los siguientes requisitos instalados:

Python 3.12.4
C++ compiler (necesario para algunas dependencias)

# Puedes instalar las dependencias utilizando el archivo requirements.txt. Las dependencias necesarias son:

# fastapi
## uvicorn
### numpy
scikit-learn
tensorflow
keras
pillow
matplotlib
python-multipart

Pasos para Configurar el Entorno

Crear un entorno virtual (opcional pero recomendado):

python3 -m venv venv (o python -m venv venv)

Activar el entorno virtual:

En Windows:

.\venv\Scripts\activate

En Linux/Mac:

source venv/bin/activate

Instalar las dependencias:


pip install -r ../requirements.txt
Instalar los compiladores de C++ si es necesario (dependiendo del sistema operativo).

Ejecutar la API:

Una vez que el entorno esté configurado, ejecuta la aplicación usando uvicorn:

"AVISO: Debe ejecutarse desde la carpeta 'app' como directorio raíz y dentro del entorno creado."

uvicorn main:app --reload
Esto iniciará la API en el servidor local y podrás acceder a ella en http://localhost:8000.

Endpoints de la API

POST /compare_image/

Este endpoint recibe una imagen en formato multipart/form-data y devuelve los 5 productos más similares al recibido en función de sus embeddings.

Parámetros:

file: Archivo de imagen que el usuario quiere comparar.

Respuesta:

Devuelve un JSON con los 5 productos más similares:

{
    "top_similar_products": ["id_1", "id_2", "id_3", "id_4", "id_5"]
}

Explicación del Código

Archivos Principales

main.py: Contiene la lógica de la API. El endpoint /compare_image/ recibe una imagen, extrae su embedding utilizando el modelo de red neuronal y la compara con los embeddings almacenados en un archivo JSON (image_embeddings.json).

embeddings.py: Contiene funciones que se encargan de procesar imágenes, calcular embeddings, comparar similitudes y cargar datos de un archivo JSON.

Funciones Importantes

image_to_embedding: Esta función procesa la imagen subida por el usuario, la redimensiona y la convierte en un embedding utilizando el modelo entrenado.

compare_embeddings: Compara los embeddings de dos imágenes utilizando la similitud de coseno.

get_embeddings_from_json: Carga los embeddings y sus identificadores desde el archivo image_embeddings.json.

Modelo

El modelo utilizado para extraer los embeddings es un modelo finetuneado basado en InceptionV3 que fue entrenado sobre el dataset DeepFashion.

Cargar el modelo:

modelo_embeddings = load_model('model/Modelo_final_embenddings_ropa.keras')

Este modelo toma imágenes de ropa y genera un vector (embedding) que representa características importantes de la imagen.

Cómo Funciona

Cuando un usuario sube una imagen, esta es procesada por la función image_to_embedding para generar su embedding.
Los embeddings de las imágenes almacenadas en el archivo image_embeddings.json se comparan con el embedding de la imagen subida utilizando la función compare_embeddings.
Se ordenan los resultados por similitud y se devuelven los 5 productos más similares.

Contribuciones
Si deseas contribuir a este proyecto, puedes hacer un fork de este repositorio, realizar cambios o mejoras y luego enviar un pull request. Las contribuciones son siempre bienvenidas.

Licencia
Este proyecto está licenciado bajo la MIT License.

Si tienes alguna pregunta o necesitas más detalles, no dudes en abrir un issue en el repositorio o contactar con el desarrollador.
