import numpy as np
from scipy.ndimage import interpolation
import math

# berechnung von winkel
# identifikation von ganzzyklen
# abstände
# lineare interpolation and dtw

def get_angle(point_1, point_2, point_3, which_part):
    """
    Calculate angles with the keypoints provided as an numpay array. The calculation of
    the angle depends on the body part that is under analysis.
    """
    # TODO: add which_part == 'elbow'
    # TODO: raise warning

    if point_1[0] == None or point_1[1] == None or point_2[0] == None or point_2[1] == None or point_3[0] == None or point_3[1] == None:
        angle = None

    elif point_1[0] == 0 and point_1[1] == 0:
        angle = 0

    elif point_2[0] == 0 and point_2[1] == 0:
        angle = 0

    elif point_3[0] == 0 and point_3[1] == 0:
        angle = 0

    else:

        v1 = point_1 - point_2
        v2 = point_3 - point_2

        if which_part == 'knee':
            reference = (-v1 / np.linalg.norm(v1)) * np.linalg.norm(v2)

            # calculate the angle between the two vectors
            raw_angle = np.arccos(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

            # calculate the extension/flexion by comparing the positon of the vectors
            if v2[1] > reference[1]:
                angle = np.rad2deg(raw_angle - np.pi)

            else:
                angle = np.rad2deg(np.pi - raw_angle)

        if which_part == 'hip':
            reference = (-v1 / np.linalg.norm(v1)) * np.linalg.norm(v2)

            # calculate the angle between the two vectors
            raw_angle = np.arccos(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

            # calculate the extension/flexion by comparing the positon of the vectors
            if v2[0] > reference[0]:
                angle = np.rad2deg(raw_angle - np.pi)

            else:
                angle = np.rad2deg(np.pi - raw_angle)

        if which_part == 'ankle':
            # calculate the extension/flexion of the ankle angle
            reference = v1[1], -v1[0]
            reference = (reference / np.linalg.norm(v1)) * np.linalg.norm(v2)

            raw_angle = np.arccos(np.dot(reference, v2) /
                                  (np.linalg.norm(reference) * np.linalg.norm(v2)))

            if v2[1] < reference[1]:
                angle = -np.rad2deg(raw_angle)
            else:
                angle = np.rad2deg(raw_angle)

        if which_part == 'other':

            radian_angle = np.arccos(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

            angle = np.rad2deg(radian_angle)

    return angle


def detect_heel_strikes(angles):
    """
    Find heel strikes in the joint angles timeseries by using derivation
    """
    # TODO: think about ways to use the ankle angles
    a = 0
    indexes = []

    grad = np.gradient(angles)
    for index, js in enumerate(grad):
        if grad[index] > 0:
            a = index
        # has to be smalle than -0.4 so that inaccuracies can be compensated
        if a != 0 and grad[index] < -0.8:
            indexes.insert(index, a + 1)
            a = 0
    
    if len(indexes) < 2:
        indexes.clear()
        a = 0
        for index, js in enumerate(grad):
            if grad[index] > 0:
                a = index
            # has to be smalle than -0.4 so that inaccuracies can be compensated
            if a != 0 and grad[index] < -0.3:
                indexes.insert(index, a + 1)
                a = 0
    
    if len(indexes) < 2:
        indexes.clear()
        a = 0
        for index, js in enumerate(grad):
            if grad[index] > 0:
                a = index
            # has to be smalle than -0.4 so that inaccuracies can be compensated
            if a != 0 and grad[index] < 0:
                indexes.insert(index, a + 1)
                a = 0
    return indexes


def get_cadence(beginn_index, end_index):
    """
    Calculate the cadence by using the length of a gait cylce. The frame rate of the videos is 50 frames/second
    """
    if beginn_index < end_index:
        frames = end_index - beginn_index
        double_step_duration = frames / 50
        double_steps_per_minute = 60 / double_step_duration
        cadence = double_steps_per_minute * 2
    else:
        cadence = 0
    
    return cadence


def get_pixel_distance(points_x, points_y, beginn_index, end_index):
    """
    Get the pixel distance of the body keypoints from begin to the end of a gait cycle.
    """
    beginn_point_x = points_x[beginn_index]
    beginn_point_y = points_y[beginn_index]
    beginn_point = np.array([beginn_point_x, beginn_point_y])

    end_point_x = points_x[end_index]
    end_point_y = points_y[end_index]
    end_point = np.array([end_point_x, end_point_y])

    distance = np.linalg.norm(beginn_point - end_point)
    return distance



def linear_interpolate_angles(angles):
    """
    Resize the array with angles to a shape of 0 to 100 for normalizing the gait cycle for different length.
    """
    
    new_size = 101
    factor = new_size / len(angles)
    angles_resized = interpolation.zoom(angles, factor)

    return angles_resized
