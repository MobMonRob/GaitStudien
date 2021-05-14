from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.metrics import f1_score
import lightgbm as lgb

#TODO: think about data normalization

def split_training_test_data(data_x, data_y):
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)    
    return x_train, x_test, y_train, y_test

#TODO: try out different parameters
def train_classifier(x_train, y_train, method):
    if method == "knn":
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(x_train, y_train)
        return clf
    if method == "svm":
        clf = SVC()
        clf.fit(x_train, y_train)
        return clf
    if method == "naive_bayes":
        clf = MultinomialNB()
        clf.fit(x_train, y_train)
        return clf
    if method == "random_forest":
        clf = RandomForestClassifier(n_estimators = 10000)
        clf.fit(x_train, y_train)
        return clf

def evaluate_classification_results(y_true, y_pred, average):
    return f1_score(y_true, y_pred, average=average)