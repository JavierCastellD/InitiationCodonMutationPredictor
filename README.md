# Predictor Mutacion Codón Inicio
Proyecto de investigación y TFG sobre la creación de un predictor de mutaciones en el codón de inicio. 
Grado en Ingeniería Informática UMU. Curso 2019-2020.

## Estructura
### src/
En src/ se encuentra el código que hemos ido desarrollando y se separa en las siguientes carpetas:
- machine_learning: es donde están los scripts que se han utilizado para las pruebas de Machine Learning.
- web: es donde están los ficheros relacionados con la página web.

#### machine_learning/
En esta carpeta nos encontramos los siguientes ficheros:
- DL.py: es donde se hacen las pruebas con redes neuronales. [WIP]
- HT.py: es donde se hacen las pruebas de hiperparámetros para un modelo concreto.
- HTEnsemble.py: es donde se hacen las pruebas de hiperparámetros para un modelo ensemble.
- HTIndv.py: es donde se hacen las pruebas de hiperparámetros de forma individual, generándose también gráficas para Accuracy, ROC_AUC y Specificity
- ML.py: es donde se hacen pruebas con distintos algoritmos de Machine Learning para obtener los valores en función del % de UnderSampling aplicado, así como número de features.
- MLClustering.py: similar a ML.py pero para Clustering.
- TestModelo.py: script para obtener métricas algo más fiables sobre un modelo.
- persistModelo.py: script para persistir un modelo y su codificador.
- predictorcodoninicio.py: fichero con distintas funciones que se han ido utilizando.

#### web/
En esta carpeta nos encontramos los siguientes ficheros:
- controller.js: Controlador JS de la web. [WIP]
- index.html: Fichero HTML de la web. [WIP]
- index.py: Endpoint que recoge las peticiones del controlador. [WIP]

### data/
En data/ es donde se almacenan los ficheros de entrada y salida, ordenados en sus respectivas carpetas. Dentro de la carpeta de salida tenemos una carpeta para cada uno de los scripts anteriores que generen archivos. Asimismo, hay una carpeta con los modelos que se han ido generando.
