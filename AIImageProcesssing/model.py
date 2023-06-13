from sklearn import model_selection
from cnn import CNNClassifier
from dataLoad import categorical_to_numpy, load_data, plot_one_image, plot_acc


'''
def model_to_string(model):
    import re
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    sms = "\n".join(stringlist)
    sms = re.sub('_\d\d\d','', sms)
    sms = re.sub('_\d\d','', sms)
    sms = re.sub('_\d','', sms)  
    return sms'''







data_raw, labels_raw = load_data()
data = data_raw.astype(float)
labels = categorical_to_numpy(labels_raw)
inputs_train, inputs_test, labels_train, labels_test = model_selection.train_test_split(data, labels, test_size=0.2, random_state=1)



#plot_one_image(data_raw, labels_raw, 300)


cnn = CNNClassifier(inputs_train, labels_train, inputs_test, labels_test)