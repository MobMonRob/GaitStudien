import pandas as pd
import numpy as np
import scipy.ndimage
import data_reader as dr
import gait_utils as gu
import os

def write_subfolders_to_csv(path_to_folder):
    directory_contents = os.listdir(path_to_folder)
    # write csv file with gait features for each subfolder in path_to_folder directory
    for item in directory_contents:
        # check if current item is a directory
        if os.path.isdir(path_to_folder + item):
            # write csv file with gait features for the current subfolder
            write_folder_to_csv(path_to_folder + item)

def write_folder_to_csv(path_to_folder):
    df_gait_features = extract_gait_features(path_to_folder)
    if df_gait_features is not None:
        df_gait_features.to_csv (path_to_folder + ".csv", sep = ';', index = False, header=True)
    else: 
        print("Error while trying to extract gait features for folder " + path_to_folder)

def extract_gait_features(keypoints_folder):
    
    keypoints_dataframe = dr.read_data_from_session(keypoints_folder)
    df_angles = extract_angles(keypoints_dataframe)
    
    indexes = get_gait_cycle_indexes(df_angles["right_hip_angles"])
    
    # check if a gait cycle could be detected
    if len(indexes) > 1:

        #TODO: remove code snippet
        if indexes[1] - indexes[0] < 40:
            print(keypoints_folder)
            print(indexes)
    
        df_angles_gaitcycle = get_gait_cycle_angles(df_angles, indexes)

        cadence = gu.get_cadence(indexes[0], indexes[1])
        df_angles_gaitcycle["cadence"] = cadence

        pixel_distance = gu.get_pixel_distance(keypoints_dataframe["heel_r_x"], keypoints_dataframe["heel_r_y"], indexes[0], indexes[1])
        df_angles_gaitcycle["pixel_distance"] = pixel_distance
    
        df_angles_gaitcycle["begin_index"] = indexes[0]
        df_angles_gaitcycle["end_index"] = indexes[1]
        return df_angles_gaitcycle
    

    else:
        # if no gait cycle could be detected return None
        # TODO: improve exception handling
        return None

def get_gait_cycle_angles(df_angles, indexes):
    """
    Get the angles during the gait cycle
    """

    begin_gaitcycle = indexes[0]
    # make sure that the end of the gaitcycle is also included in the data
    end_gaitcycle = indexes[1] + 1

    if begin_gaitcycle < end_gaitcycle:

        right_knee_angles = df_angles["right_knee_angles"]
        left_knee_angles = df_angles["left_knee_angles"]
        right_hip_angles = df_angles["right_hip_angles"]
        left_hip_angles = df_angles["left_hip_angles"]
        right_midhip_angles = df_angles["right_midhip_angles"]
        left_midhip_angles = df_angles["left_midhip_angles"]
        right_ankle_angles = df_angles["right_ankle_angles"]
        left_ankle_angles = df_angles["left_ankle_angles"]
        right_elbow_angles = df_angles["right_elbow_angles"]
        left_elbow_angles = df_angles["left_elbow_angles"]
        neck_angles = df_angles["neck_angles"]

        right_knee_angles_gaitcycle = right_knee_angles[begin_gaitcycle:end_gaitcycle]
        left_knee_angles_gaitcycle = left_knee_angles[begin_gaitcycle:end_gaitcycle]
        right_hip_angles_gaitcycle = right_hip_angles[begin_gaitcycle:end_gaitcycle]
        left_hip_angles_gaitcycle = left_hip_angles[begin_gaitcycle:end_gaitcycle]
        right_midhip_angles_gaitcycle = right_midhip_angles[begin_gaitcycle:end_gaitcycle]
        left_midhip_angles_gaitcycle = left_midhip_angles[begin_gaitcycle:end_gaitcycle]
        right_ankle_angles_gaitcycle = right_ankle_angles[begin_gaitcycle:end_gaitcycle]
        left_ankle_angles_gaitcycle = left_ankle_angles[begin_gaitcycle:end_gaitcycle]
        right_elbow_angles_gaitcycle = right_elbow_angles[begin_gaitcycle:end_gaitcycle]
        left_elbow_angles_gaitcycle = left_elbow_angles[begin_gaitcycle:end_gaitcycle]
        neck_angles_gaitcycle = neck_angles[begin_gaitcycle:end_gaitcycle]

        df_rha = pd.DataFrame(right_hip_angles_gaitcycle,
                              columns=["right_hip_angles"])
        df_rka = pd.DataFrame(right_knee_angles_gaitcycle,
                              columns=["right_knee_angles"])
        df_raa = pd.DataFrame(right_ankle_angles_gaitcycle, columns=[
                              "right_ankle_angles"])
        df_rea = pd.DataFrame(right_elbow_angles_gaitcycle, columns=[
                              "right_elbow_angles"])
        df_rma = pd.DataFrame(right_midhip_angles_gaitcycle, columns=[
                              "right_midhip_angles"])

        df_lha = pd.DataFrame(left_hip_angles_gaitcycle,
                              columns=["left_hip_angles"])
        df_lke = pd.DataFrame(left_knee_angles_gaitcycle,
                              columns=["left_knee_angles"])
        df_laa = pd.DataFrame(left_ankle_angles_gaitcycle,
                              columns=["left_ankle_angles"])
        df_lea = pd.DataFrame(left_elbow_angles_gaitcycle,
                              columns=["left_elbow_angles"])
        df_lma = pd.DataFrame(left_midhip_angles_gaitcycle, columns=[
                              "left_midhip_angles"])

        df_neck = pd.DataFrame(neck_angles_gaitcycle, columns=["neck_angles"])

        df_angles_gaitcycle = df_rha.join(
            [df_rka, df_raa, df_rea, df_rma, df_lha, df_lke, df_laa, df_lea, df_lma, df_neck])

        return df_angles_gaitcycle
    else:
        print("Problems with indexes of the gaitcycle!")


def get_gait_cycle_indexes(right_hip_angles):
    """
    Get the beginning and start index of a gait cycle by using the indexes of detected heel strikes. For better results use the gaussian_filter before with sigma 5.
    """
    # TODO: try out to identify gait cycles by using the right ankle angles
    indexes = []
    filtered_angles = scipy.ndimage.gaussian_filter(right_hip_angles, sigma=5)
    heelstrikes = gu.detect_heel_strikes(filtered_angles)
    if len(heelstrikes) > 1:
        indexes.append(heelstrikes[0])
        indexes.append(heelstrikes[1])
        if heelstrikes[0] - heelstrikes[1] < 35 and len(heelstrikes) > 2:
            indexes.clear()
            indexes.append(heelstrikes[0])
            indexes.append(heelstrikes[2])
    else:
        print("No double step detected!")
    return indexes


def extract_angles(keypoints_dataframe):
    """
    Calculate the angles with the body keypoints of the datframe
    """

    right_knee_angles = []
    left_knee_angles = []
    right_hip_angles = []
    left_hip_angles = []
    right_midhip_angles = []
    left_midhip_angles = []
    right_ankle_angles = []
    left_ankle_angles = []
    right_elbow_angles = []
    left_elbow_angles = []
    neck_angles = []

    for index, row in keypoints_dataframe.iterrows():
        head = np.array([row["head_x"], row["head_y"]])
        neck = np.array([row["neck_x"], row["neck_y"]])
        shoulder_r = np.array([row["shoulder_r_x"], row["shoulder_r_y"]])
        elbow_r = np.array([row["elbow_r_x"], row["elbow_r_y"]])
        wrist_r = np.array([row["wrist_r_x"], row["wrist_r_y"]])
        shoulder_l = np.array([row["shoulder_l_x"], row["shoulder_l_y"]])
        elbow_l = np.array([row["elbow_l_x"], row["elbow_l_y"]])
        wrist_l = np.array([row["wrist_l_x"], row["wrist_l_y"]])
        midhip = np.array([row["midhip_x"], row["midhip_y"]])
        hip_r = np.array([row["hip_r_x"], row["hip_r_y"]])
        knee_r = np.array([row["knee_r_x"], row["knee_r_y"]])
        ankle_r = np.array([row["ankle_r_x"], row["ankle_r_y"]])
        hip_l = np.array([row["hip_l_x"], row["hip_l_y"]])
        knee_l = np.array([row["knee_l_x"], row["knee_l_y"]])
        ankle_l = np.array([row["ankle_l_x"], row["ankle_l_y"]])
        bigtoe_l = np.array([row["bigtoe_l_x"], row["bigtoe_l_y"]])
        heel_l = np.array([row["heel_l_x"], row["heel_l_y"]])
        bigtoe_r = np.array([row["bigtoe_r_x"], row["bigtoe_r_y"]])
        heel_r = np.array([row["heel_r_x"], row["heel_r_y"]])

        rha = 180 - gu.get_angle(neck, hip_r, knee_r, "hip")  # right hip angle
        rke = 180 - gu.get_angle(hip_r, knee_r, ankle_r,
                                 "knee")  # right knee angle
        raa = 90 - gu.get_angle(knee_r, ankle_r, bigtoe_r,
                                "ankle")  # right ankle angle
        rea = gu.get_angle(shoulder_r, elbow_r, wrist_r,
                           "other")  # right elbow angle
        rma = gu.get_angle(midhip, hip_r, knee_r,
                           "other")  # right midhip angle

        lha = 180 - gu.get_angle(neck, hip_l, knee_l, "hip")  # left hip angle
        lke = 180 - gu.get_angle(hip_l, knee_l, ankle_l,
                                 "knee")  # left knee angle
        laa = 90 - gu.get_angle(knee_l, ankle_l, bigtoe_l,
                                "ankle")  # left ankle angle
        lea = gu.get_angle(shoulder_l, elbow_l, wrist_l,
                           "other")  # left elbow angle
        lma = gu.get_angle(midhip, hip_l, knee_l, "other")  # left midhip angle

        na = gu.get_angle(head, neck, midhip, "other")  # neck angle

        right_hip_angles.append(rha)
        right_knee_angles.append(rke)
        right_ankle_angles.append(raa)
        right_elbow_angles.append(rea)
        right_midhip_angles.append(rma)

        left_hip_angles.append(lha)
        left_knee_angles.append(lke)
        left_ankle_angles.append(laa)
        left_elbow_angles.append(lea)
        left_midhip_angles.append(lma)

        neck_angles.append(na)

    df_rha = pd.DataFrame(right_hip_angles, columns=["right_hip_angles"])
    df_rka = pd.DataFrame(right_knee_angles, columns=["right_knee_angles"])
    df_raa = pd.DataFrame(right_ankle_angles, columns=["right_ankle_angles"])
    df_rea = pd.DataFrame(right_elbow_angles, columns=["right_elbow_angles"])
    df_rma = pd.DataFrame(right_midhip_angles, columns=["right_midhip_angles"])

    df_lha = pd.DataFrame(left_hip_angles, columns=["left_hip_angles"])
    df_lke = pd.DataFrame(left_knee_angles, columns=["left_knee_angles"])
    df_laa = pd.DataFrame(left_ankle_angles, columns=["left_ankle_angles"])
    df_lea = pd.DataFrame(left_elbow_angles, columns=["left_elbow_angles"])
    df_lma = pd.DataFrame(left_midhip_angles, columns=["left_midhip_angles"])

    df_neck = pd.DataFrame(neck_angles, columns=["neck_angles"])

    df_angles = df_rha.join(
        [df_rka, df_raa, df_rea, df_rma, df_lha, df_lke, df_laa, df_lea, df_lma, df_neck])
    return df_angles