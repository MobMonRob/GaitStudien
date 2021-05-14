import os
import json
import pandas as pd
import numpy as np
import warnings

HEAD = 0
NECK = 1
SHOULDER_R = 2
ELBOW_R = 3
WRIST_R = 4
SHOULDER_L = 5
ELBOW_L = 6
WRIST_L = 7
MIDHIP = 8
HIP_R = 9
KNEE_R = 10
ANKLE_R = 11
HIP_L = 12
KNEE_L = 13
ANKLE_L = 14
EYE_R = 15
EYE_L = 16
EAR_R = 17
EAR_L = 18
BIGTOE_L = 19
SMALLTOE_L = 20
HEEL_L = 21
BIGTOE_R = 22
SMALLTOE_R = 23
HEEL_R = 24

def read_data_from_session(path_to_data_files):
    """
    Read the OpenPose data and select the joint points for further analysis.
    """

    # lists to store body keypoints
    head = []
    neck = []
    shoulder_r = []
    elbow_r = []
    wrist_r = []
    shoulder_l = []
    elbow_l = []
    wrist_l = []
    midhip = []
    hip_r = []
    knee_r = []
    ankle_r = []
    hip_l = []
    knee_l = []
    ankle_l = []
    eye_r = []
    eye_l = []
    ear_r = []
    ear_l = []
    bigtoe_l = []
    smalltoe_l = []
    heel_l = []
    bigtoe_r = []
    smalltoe_r = []
    heel_r = []

    # get all json file from folder
    json_files = [pos_json for pos_json in sorted(os.listdir(
        path_to_data_files)) if pos_json.endswith('.json')]

    # read the body keypoints of each frame and add them to the list
    for json_file in json_files:
        f = open(os.path.join(path_to_data_files, json_file), 'r')
        data = f.read()
        json_data = json.loads(data)

        # get values of body keypoints and add them to the list
        head.append(get_keypoint_from_json(json_data, HEAD))
        neck.append(get_keypoint_from_json(json_data, NECK))
        shoulder_r.append(get_keypoint_from_json(json_data, SHOULDER_R))
        elbow_r.append(get_keypoint_from_json(json_data, ELBOW_R))
        wrist_r.append(get_keypoint_from_json(json_data, WRIST_R))
        shoulder_l.append(get_keypoint_from_json(json_data, SHOULDER_L))
        elbow_l.append(get_keypoint_from_json(json_data, ELBOW_L))
        wrist_l.append(get_keypoint_from_json(json_data, WRIST_L))
        midhip.append(get_keypoint_from_json(json_data, MIDHIP))
        hip_r.append(get_keypoint_from_json(json_data, HIP_R))
        knee_r.append(get_keypoint_from_json(json_data, KNEE_R))
        ankle_r.append(get_keypoint_from_json(json_data, ANKLE_R))
        hip_l.append(get_keypoint_from_json(json_data, HIP_L))
        knee_l.append(get_keypoint_from_json(json_data, KNEE_L))
        ankle_l.append(get_keypoint_from_json(json_data, ANKLE_L))
        eye_r.append(get_keypoint_from_json(json_data, EYE_R))
        eye_l.append(get_keypoint_from_json(json_data, EYE_L))
        ear_r.append(get_keypoint_from_json(json_data, EAR_R))
        ear_l.append(get_keypoint_from_json(json_data, EAR_L))
        bigtoe_l.append(get_keypoint_from_json(json_data, BIGTOE_L))
        smalltoe_l.append(get_keypoint_from_json(json_data, SMALLTOE_L))
        heel_l.append(get_keypoint_from_json(json_data, HEEL_L))
        bigtoe_r.append(get_keypoint_from_json(json_data, BIGTOE_R))
        smalltoe_r.append(get_keypoint_from_json(json_data, SMALLTOE_R))
        heel_r.append(get_keypoint_from_json(json_data, HEEL_R))

    # store the lists of body keypoints in a dataframe for further analysis
    df_head = pd.DataFrame(head, columns=['head_x', 'head_y'], dtype=float)
    df_neck = pd.DataFrame(neck, columns=['neck_x', 'neck_y'], dtype=float)
    df_shoulder_r = pd.DataFrame(
        shoulder_r, columns=['shoulder_r_x', 'shoulder_r_y'], dtype=float)
    df_elbow_r = pd.DataFrame(
        elbow_r, columns=['elbow_r_x', 'elbow_r_y'], dtype=float)
    df_wrist_r = pd.DataFrame(
        wrist_r, columns=['wrist_r_x', 'wrist_r_y'], dtype=float)
    df_shoulder_l = pd.DataFrame(
        shoulder_l, columns=['shoulder_l_x', 'shoulder_l_y'], dtype=float)
    df_elbow_l = pd.DataFrame(
        elbow_l, columns=['elbow_l_x', 'elbow_l_y'], dtype=float)
    df_wrist_l = pd.DataFrame(
        wrist_l, columns=['wrist_l_x', 'wrist_l_y'], dtype=float)
    df_midhip = pd.DataFrame(
        midhip, columns=['midhip_x', 'midhip_y'], dtype=float)
    df_hip_r = pd.DataFrame(hip_r, columns=['hip_r_x', 'hip_r_y'], dtype=float)
    df_knee_r = pd.DataFrame(
        knee_r, columns=['knee_r_x', 'knee_r_y'], dtype=float)
    df_ankle_r = pd.DataFrame(
        ankle_r, columns=['ankle_r_x', 'ankle_r_y'], dtype=float)
    df_hip_l = pd.DataFrame(hip_l, columns=['hip_l_x', 'hip_l_y'], dtype=float)
    df_knee_l = pd.DataFrame(
        knee_l, columns=['knee_l_x', 'knee_l_y'], dtype=float)
    df_ankle_l = pd.DataFrame(
        ankle_l, columns=['ankle_l_x', 'ankle_l_y'], dtype=float)
    df_eye_r = pd.DataFrame(eye_r, columns=['eye_r_x', 'eye_r_y'], dtype=float)
    df_eye_l = pd.DataFrame(eye_l, columns=['eye_l_x', 'eye_l_y'], dtype=float)
    df_ear_r = pd.DataFrame(ear_r, columns=['ear_r_x', 'ear_r_y'], dtype=float)
    df_ear_l = pd.DataFrame(ear_l, columns=['ear_l_x', 'ear_l_y'], dtype=float)
    df_bigtoe_l = pd.DataFrame(
        bigtoe_l, columns=['bigtoe_l_x', 'bigtoe_l_y'], dtype=float)
    df_smalltoe_l = pd.DataFrame(
        smalltoe_l, columns=['smalltoe_l_x', 'smalltoe_l_y'], dtype=float)
    df_heel_l = pd.DataFrame(
        heel_l, columns=['heel_l_x', 'heel_l_y'], dtype=float)
    df_bigtoe_r = pd.DataFrame(
        bigtoe_r, columns=['bigtoe_r_x', 'bigtoe_r_y'], dtype=float)
    df_smalltoe_r = pd.DataFrame(
        smalltoe_r, columns=['smalltoe_r_x', 'smalltoe_r_y'], dtype=float)
    df_heel_r = pd.DataFrame(
        heel_r, columns=['heel_r_x', 'heel_r_y'], dtype=float)

    df = df_head.join([df_neck, df_shoulder_r, df_elbow_r, df_wrist_r, df_shoulder_l, df_elbow_l, df_wrist_l, df_midhip, df_hip_r, df_knee_r, df_ankle_r,
                      df_hip_l, df_knee_l, df_ankle_l, df_eye_r, df_eye_l, df_ear_r, df_ear_l, df_bigtoe_l, df_smalltoe_l, df_heel_l, df_bigtoe_r, df_smalltoe_r, df_heel_r])
    return df

def get_keypoint_from_json(json_data, keypoint):
    """
    Read the Keypoints from json data. If multiple people are detected, it returns the keypoint position of the first person and prints a warning.
    """
    # warn the user if multiple persons are detected on a frame
    if(len(json_data["people"]) > 1):
        warnings.warn(
            "More than one person was detected on the keypoints of the video.")

    # get the lists of 2d body keypoints
    body_keypoints = json_data["people"][0]["pose_keypoints_2d"]

    # select the right body keypoints
    keypoint_x = body_keypoints[keypoint * 3]
    keypoint_y = body_keypoints[keypoint * 3 + 1]
    # TODO: think about linear interpolation for improving missing points

    # create arrays to store pairs of the position
    return np.array([keypoint_x, keypoint_y])


def read_data_from_csv_files(path_to_folder):
    """
    Read the 
    """
    csv_files = [pos_csv for pos_csv in sorted(os.listdir(
        path_to_folder)) if pos_csv.endswith('.csv')]
    
    data_x = []
    data_y = []
    
    for csv_file in csv_files:
        data = pd.read_csv(path_to_folder + csv_file, sep = ';')
        data_x.append(data)
        data_y.append(get_person_from_csv_file(csv_file))
   
    return data_x, data_y

def get_person_from_csv_file(filename):
    parts = filename.split("_")
    person = parts[0] + "_" + parts[1] + "_" + parts[2] + "_" + parts[3]
    return person
