from matcher import *

data_object = Data(api_key="")
classifier = Classifier(data_object, train=False, load_job_ads=True)

classifier.training(vocab_size=4000)
result = classifier.predict_top_n_classes('Wir suchen einen Data Scientist/Modellierer/Statistiker', 10)

print(result)