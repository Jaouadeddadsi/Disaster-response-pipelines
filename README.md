# Disaster Response Pipeline

This project was designed to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) in order to build a model for an API that classifies disaster messages.

 The project contain a machine learning pipeline that categorize a message sent during disaster event so we can send the message to an appropriate disaster relief agency.

 The project include  a web app where an emergency worker can input a new message and get classification results in several categories. The web app  display also visualizations of the data.  


##  Installations

This project requires **Python 3.x** and the following Python libraries installed:

- [Pandas](http://pandas.pydata.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [Nltk](https://www.nltk.org/)
- [Flask](http://flask.pocoo.org/)
- [Bootstrap](https://getbootstrap.com/)
- [Plotly](https://plot.ly/)



## Files Descriptions
**app**
- template :
 - master.html: main page of web app
 - go.html: classification result page of web app

- run.py : Flask file that runs app

**data**
- disaster_messages.csv: file contains massages
- disaster_categories.csv: file contains messages categories
- process_data.py: ETL pipline to process data
- DisasterResponse.db: database to save clean data to

**models**
- train_classifier.py : Ml pipeline to train the model and save it into pkl file
- classifier.pkl: saved model

**README.md**


## Running codes
- Process data:
  ` python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db `

- Train classifier:
`python train_classifier.py ../data/DisasterResponse.db classifier.pkl` .
Running train_classifier.py take a long time because of the **Gridsearchcv**. for this project I run it in **GCP**.

- Web App :
`python run.py`


## Author

-   **Jaouad Eddadsi**   [linkedin](https://www.linkedin.com/in/jaouad-eddadsi-01bb34163/)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

I would like to thank [Udacity](https://eu.udacity.com/) for this amazing project, and [Figure Eight](https://www.figure-eight.com/)  for providing the data.
