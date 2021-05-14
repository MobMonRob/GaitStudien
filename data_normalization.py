import gait_utils as gu
import data_reader as dr
import numpy as np
import pandas as pd
import scipy

#TODO: think about preprocessing with sklearn

def get_and_normalize_data(path_to_folder):
    data_x, data_y = dr.read_data_from_csv_files(path_to_folder)
    data_x_interpolated = interpolate_list_of_dataframes(data_x, "before")
    
    return data_x_interpolated, data_y

def convert_dataframes_to_2d_array(dataframes, features):
    data = []
    for dataframe in dataframes:
        dataframe_array = convert_dataframe_to_numpy_array(dataframe, features)
        data.append(dataframe_array)
    data_as_numpy_array = np.array(data)
    return data_as_numpy_array

def convert_dataframe_to_numpy_array(dataframe, features):
    data_as_list = []
    for feature in features:
        if feature == "right_hip":
            right_hip_angles = dataframe['right_hip_angles'].tolist()
            data_as_list.extend(right_hip_angles)
        if feature == "left_hip":
            left_hip_angles = dataframe["left_hip_angles"].tolist()
            data_as_list.extend(left_hip_angles)
        if feature == "right_knee":
            right_knee_angles = dataframe["right_knee_angles"].tolist()
            data_as_list.extend(right_knee_angles)
        if feature == "left_ankle":
            left_knee_angles = dataframe["left_knee_angles"].tolist()
            data_as_list.extend(left_knee_angles)
        if feature == "right_ankle":
            right_ankle = dataframe["right_ankle_angles"].tolist()
            data_as_list.extend(right_ankle)
        if feature == "left_ankle":
            left_ankle = dataframe["left_ankle_angles"].tolist()
            data_as_list.extend(left_ankle)
        if feature == "right_elbow":
            right_elbow = dataframe["right_elbow_angles"].tolist()
            data_as_list.extend(right_elbow)
        if feature == "left_elbow":
            left_elbow = dataframe["left_elbow_angles"].tolist()
            data_as_list.extend(left_elbow)
        if feature == "right_midhip":
            right_midhip = dataframe["right_midhip_angles"].tolist()
            data_as_list.extend(right_midhip)
        if feature == "left_midhip":
            left_midhip = dataframe["left_midhip_angles"].tolist()
            data_as_list.extend(left_midhip)
        if feature == "neck":
            neck = dataframe["neck_angles"].tolist()
            data_as_list.extend(neck)
        if feature == "cadence":
            cadence = dataframe["cadence"][1]
            data_as_list.append(cadence)
        if feature == "pixel_distance":
            pixel_distance = dataframe["pixel_distance"][1]
            data_as_list.append(pixel_distance)
    data_as_numpy_array = np.array(data_as_list)
    return data_as_numpy_array

def interpolate_list_of_dataframes(dataframes, option):
    dataframes_interpolated = []
    if option == "before":
        for dataframe in dataframes:
            dataframe_filtered = use_gaussian_filter(dataframe, 5)
            dataframe_interpolated = linear_interpolate_dataframe(dataframe_filtered)
            dataframes_interpolated.append(dataframe_interpolated)
    if option == "after":
        for dataframe in dataframes:
            dataframe_interpolated = linear_interpolate_dataframe(dataframe)
            dataframe_filtered = use_gaussian_filter(dataframe_interpolated, 5)
            dataframes_interpolated.append(dataframe_filtered)
    if option == "without":
        for dataframe in dataframes:
            dataframe_interpolated = linear_interpolate_dataframe(dataframe)
            dataframes_interpolated.append(dataframe_interpolated)

    return dataframes_interpolated


def use_gaussian_filter(dataframe, sigma):
    # use gaussian filter

    right_knee_angles_filtered = scipy.ndimage.gaussian_filter(dataframe["right_knee_angles"], sigma=sigma)
    left_knee_angles_filtered = scipy.ndimage.gaussian_filter(dataframe["left_knee_angles"], sigma=sigma)
    right_hip_angles_filtered = scipy.ndimage.gaussian_filter(dataframe["right_hip_angles"], sigma=sigma)
    left_hip_angles_filtered = scipy.ndimage.gaussian_filter(dataframe["left_hip_angles"], sigma=sigma)
    right_midhip_angles_filtered = scipy.ndimage.gaussian_filter(dataframe["right_midhip_angles"], sigma=sigma)
    left_midhip_angles_filtered = scipy.ndimage.gaussian_filter(dataframe["left_midhip_angles"], sigma=sigma)
    right_ankle_angles_filtered = scipy.ndimage.gaussian_filter(dataframe["right_ankle_angles"], sigma=sigma)
    left_ankle_angles_filtered = scipy.ndimage.gaussian_filter(dataframe["left_ankle_angles"], sigma=sigma)
    right_elbow_angles_filtered = scipy.ndimage.gaussian_filter(dataframe["right_elbow_angles"], sigma=sigma)
    left_elbow_angles_filtered = scipy.ndimage.gaussian_filter(dataframe["left_elbow_angles"], sigma=sigma)
    neck_angles_filtered = scipy.ndimage.gaussian_filter(dataframe["neck_angles"], sigma=sigma)

    df_rha = pd.DataFrame(right_hip_angles_filtered, columns=["right_hip_angles"])
    df_rka = pd.DataFrame(right_knee_angles_filtered, columns=["right_knee_angles"])
    df_raa = pd.DataFrame(right_ankle_angles_filtered, columns=["right_ankle_angles"])
    df_rea = pd.DataFrame(right_elbow_angles_filtered, columns=["right_elbow_angles"])
    df_rma = pd.DataFrame(right_midhip_angles_filtered, columns=["right_midhip_angles"])

    df_lha = pd.DataFrame(left_hip_angles_filtered, columns=["left_hip_angles"])
    df_lke = pd.DataFrame(left_knee_angles_filtered, columns=["left_knee_angles"])
    df_laa = pd.DataFrame(left_ankle_angles_filtered, columns=["left_ankle_angles"])
    df_lea = pd.DataFrame(left_elbow_angles_filtered, columns=["left_elbow_angles"])
    df_lma = pd.DataFrame(left_midhip_angles_filtered, columns=["left_midhip_angles"])

    df_neck = pd.DataFrame(neck_angles_filtered, columns=["neck_angles"])

    df_filtered_angles = df_rha.join(
        [df_rka, df_raa, df_rea, df_rma, df_lha, df_lke, df_laa, df_lea, df_lma, df_neck])

    df_filtered_angles["cadence"] = dataframe["cadence"][1]
    df_filtered_angles["pixel_distance"] = dataframe["pixel_distance"][1]
    df_filtered_angles["begin_index"] = dataframe["begin_index"][1]
    df_filtered_angles["end_index"] = dataframe["end_index"][1]
    return df_filtered_angles

def linear_interpolate_dataframe(dataframe):
    #interpolate angles to 100
    
    right_knee_angles_interpolated = gu.linear_interpolate_angles(dataframe["right_knee_angles"])
    left_knee_angles_interpolated = gu.linear_interpolate_angles(dataframe["left_knee_angles"])
    right_hip_angles_interpolated = gu.linear_interpolate_angles(dataframe["right_hip_angles"])
    left_hip_angles_interpolated = gu.linear_interpolate_angles(dataframe["left_hip_angles"])
    right_midhip_angles_interpolated = gu.linear_interpolate_angles(dataframe["right_midhip_angles"])
    left_midhip_angles_interpolated = gu.linear_interpolate_angles(dataframe["left_midhip_angles"])
    right_ankle_angles_interpolated = gu.linear_interpolate_angles(dataframe["right_ankle_angles"])
    left_ankle_angles_interpolated = gu.linear_interpolate_angles(dataframe["left_ankle_angles"])
    right_elbow_angles_interpolated = gu.linear_interpolate_angles(dataframe["right_elbow_angles"])
    left_elbow_angles_interpolated = gu.linear_interpolate_angles(dataframe["left_elbow_angles"])
    neck_angles_interpolated = gu.linear_interpolate_angles(dataframe["neck_angles"])

    df_rha = pd.DataFrame(right_hip_angles_interpolated, columns=["right_hip_angles"])
    df_rka = pd.DataFrame(right_knee_angles_interpolated, columns=["right_knee_angles"])
    df_raa = pd.DataFrame(right_ankle_angles_interpolated, columns=["right_ankle_angles"])
    df_rea = pd.DataFrame(right_elbow_angles_interpolated, columns=["right_elbow_angles"])
    df_rma = pd.DataFrame(right_midhip_angles_interpolated, columns=["right_midhip_angles"])

    df_lha = pd.DataFrame(left_hip_angles_interpolated, columns=["left_hip_angles"])
    df_lke = pd.DataFrame(left_knee_angles_interpolated, columns=["left_knee_angles"])
    df_laa = pd.DataFrame(left_ankle_angles_interpolated, columns=["left_ankle_angles"])
    df_lea = pd.DataFrame(left_elbow_angles_interpolated, columns=["left_elbow_angles"])
    df_lma = pd.DataFrame(left_midhip_angles_interpolated, columns=["left_midhip_angles"])

    df_neck = pd.DataFrame(neck_angles_interpolated, columns=["neck_angles"])

    df_interpolated_angles = df_rha.join(
        [df_rka, df_raa, df_rea, df_rma, df_lha, df_lke, df_laa, df_lea, df_lma, df_neck])

    df_interpolated_angles["cadence"] = dataframe["cadence"][1]
    df_interpolated_angles["pixel_distance"] = dataframe["pixel_distance"][1]
    df_interpolated_angles["begin_index"] = dataframe["begin_index"][1]
    df_interpolated_angles["end_index"] = dataframe["end_index"][1]
    return df_interpolated_angles
