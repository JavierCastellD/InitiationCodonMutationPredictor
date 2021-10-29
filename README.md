# A prediction method for the effect of mutations in the initiation codon
Project for the creation of a method of predicting mutations in the initiation codon by using machine learning algorithms.
The dataset obtained from Ensembl that was used in this project can be found in the following link: [WIP]

## Dependencies
The following libraries are needed to execute the scripts:
- pandas >= 0.25.3
- sklearn >= 0.22.1
- imblearn >= 0.6.2
- numpy >= 1.20.2

## Structure
### src/
Here you can find the code we have developed and it is distributed in the following folders:
- machine_learning: that is where we have placed the scripts used for the tests.
- web: web page related files.

#### machine_learning/
We can find the following files:
- HT.py: script for the hyperparameter fine-tuning.
- ML.py: script intended for testing different machine learning algorithms, as well as some parameters for them, such as % of undersampling or number of features.
- TestModelo.py: script to perform multiple tests on a model.
- comparaciones.py: script to compare the performance of a model and SIFT and PolyPhen.

#### web/
In this folder we can find files related to the webpage:
- controller.js: JS Controller for the web.
- index.html: HTML file of the web.
- index.py: Endpoint that receives the petitions from the controller.

### data/
This is where we have put the input and output files in their respective folders. 
