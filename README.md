# Using Machine Learning for the prediction of mutations in the initiation codon
Project for the creation of a method of predicting mutations in the initiation codon using Machine Learning.
Final project for the Computer Engineering degree of the University of Murcia.

Proyecto de investigación y TFG sobre la creación de un predictor de mutaciones en el codón de inicio. 
Grado en Ingeniería Informática UMU. Curso 2019-2020.

## Structure
### src/
In src/ is when you can find the code we have developed an it is distributed in the following folders:
- machine_learning: that is where we have placed the scripts used for the tests.
- web: web page related files.

En src/ se encuentra el código que hemos ido desarrollando y se separa en las siguientes carpetas:
- machine_learning: es donde están los scripts que se han utilizado para las pruebas de Machine Learning.
- web: es donde están los ficheros relacionados con la página web.

#### machine_learning/
We can find the following files:
- HT.py: script for the hyperparameter fine-tuning.
- ML.py: script intended for testing of different machine learning algorithms, as well as some parameters for them, such as % of undersampling or number of features.
- TestModelo.py: script to perform multiple tests on a model.
- comparaciones.py: script to compare the performance of a model and SIFT and PolyPhen.

Aquí podemos encontrar los siguientes ficheros:
- HT.py: es donde se hacen las pruebas de hiperparámetros para un modelo concreto.
- ML.py: es donde se hacen pruebas con distintos algoritmos de Machine Learning para obtener los valores en función del % de UnderSampling aplicado, así como número de features.
- TestModelo.py: script para obtener métricas algo más fiables sobre un modelo.
- comparaciones.py: es donde se hace la comparación de rendimiento entre un modelo y SIFT y PolyPhen.

#### web/
In this folder we can find files related to the webpage:
- controller.js: JS Controller for the web.
- index.html: HTML file of the web.
- index.py: Endpoint that receives the petitions from the controller.

En esta carpeta nos encontramos los siguientes ficheros:
- controller.js: Controlador JS de la web.
- index.html: Fichero HTML de la web.
- index.py: Endpoint que recoge las peticiones del controlador.

### data/
This is where we have put the input and output files in their respective folders. 

En data/ es donde se almacenan los ficheros de entrada y salida, ordenados en sus respectivas carpetas.
