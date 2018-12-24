
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import  tree
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    classes = ['pos', 'neg']
    # Read the data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    data_dir = "./data"
    for curr_class in classes:
        dirname = os.path.join(data_dir, curr_class)
        for fname in os.listdir(dirname):
            with open(os.path.join(dirname, fname), 'r') as f:
                content = f.read()
                if fname.startswith('cv9'):
                    test_data.append(content)
                    test_labels.append(curr_class)
                else:
                    train_data.append(content)
                    train_labels.append(curr_class)
    # create feature vectors
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

    clf = tree.DecisionTreeClassifier()
    time_start = time.time()
    clf.fit(train_vectors, train_labels)
    time_train = time.time()-time_start
    prediction = clf.predict(test_vectors)
    time_predict = time.time()-time_start-time_train

    # Print results in a nice table
    print("Results for DecisionTreeClassifier: ")
    print("Training time: %fs; Prediction time: %fs" % (time_train, time_predict))
    print(classification_report(test_labels, prediction))
    print("Reviews Prediction")
    print("Accuracy:"+str(accuracy_score(test_labels, prediction)))
    # print(prediction[10] + "----" + test_data[10])
    # print("\nReviews Prediction")
    # print(prediction[100] + "----" + test_data[100])
