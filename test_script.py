import data_reader as dr
import data_normalization as dn
import extract_features as ef
import classifier as clf
from sklearn.preprocessing import MinMaxScaler

# TODO: add example for different use cases

#ef.write_subfolders_to_csv("C:/Users/d074042/Documents/Gangdaten/Keypoints/")
feature_combinations = [["right_hip", "right_knee", "right_ankle", "right_elbow", "neck"],
    ["right_hip", "right_midhip", "right_knee", "right_ankle", "right_elbow"], 
    ["right_hip", "right_midhip"],
    ["right_hip", "right_knee"],
    ["right_hip", "right_elbow"],
    ["left_hip", "left_knee", "left_ankle", "left_elbow", "neck"],
    ["left_hip", "left_midhip", "left_knee", "left_ankle", "left_elbow"],
    ["left_hip", "left_midhip"],
    ["left_hip", "left_knee"],
    ["left_hip", "left_elbow"],
    ["cadence", "pixel_distance"], 
    ["right_hip", "right_knee", "neck"],
    ["right_hip", "neck"],
    ["left_hip", "left_knee", "neck"],
    ["left_hip", "neck"],
    ["left_hip", "right_hip", "neck"],
    ["left_hip", "right_hip", "neck", "right_ellbow"],
    ["left_hip", "right_hip"],
    ["left_hip", "right_hip", "right_elbow"],
    ["neck", "right_elbow"]]

f1_scores = []

print("start training")

for feature_combination in feature_combinations:
    data_x, data_y = dn.get_and_normalize_data("C:/Users/d074042/Documents/Gangdaten/Keypoints/")
    data_x_converted = dn.convert_dataframes_to_2d_array(data_x, feature_combination)
    
    """
    # only used for naive bayes to avoid negative values in the angles
    min_max_Scaler = MinMaxScaler()
    data_x_converted = min_max_Scaler.fit_transform(data_x_converted)
    """
    x_train, x_test, y_train, y_test = clf.split_training_test_data(data_x_converted, data_y)
    classifier = clf.train_classifier(x_train, y_train, "svm") #change classficator here
    y_pred = classifier.predict(x_test)
    f1_score = clf.evaluate_classification_results(y_test, y_pred, "micro")
    f1_scores.append(f1_score)
    print(feature_combination)
    print(f1_score)
