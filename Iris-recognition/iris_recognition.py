import numpy as np
import cv2
import os
import sys
import math
import random
import pickle
import copy
import gzip
import inspect
import itertools

from matplotlib import pyplot as plt


def compare_images(filepath1, filepath2):
    print("Analysing " + filepath1)
    rois_1 = load_rois_from_image(filepath1)

    print("Analysing " + filepath2)
    rois_2 = load_rois_from_image(filepath2)
    
    get_all_matches(rois_1, rois_2, 0.8, 10, 0.15, show=True)

def compare_binfiles(bin_path1, bin_path2):
    print("Analysing " + bin_path1)
    rois_1 = load_rois_from_bin(bin_path1)
    
    print("Analysing " + bin_path2)
    rois_2 = load_rois_from_bin(bin_path2)

    get_all_matches(rois_1, rois_2, 0.88, 10, 0.07, show=True)

def load_rois_from_image(filepath):
    img = load_image(filepath, show=True)

    print("Getting iris boundaries..")
    pupil_circle, ext_iris_circle = get_iris_boundaries(img, show=True)
    if pupil_circle is None or ext_iris_circle is None:
        print("Error finding iris boundaries!")
        return None

    print("Equalizing histogram ..")
    roi = get_equalized_iris(img, ext_iris_circle, pupil_circle, show=True)

    print("Getting roi iris images ...")
    rois = get_rois(roi, pupil_circle, ext_iris_circle, show=True)

    print("Searching for keypoints ... \n")
    sift = cv2.xfeatures2d.SIFT_create()
    load_keypoints(sift, rois, show=True)
    load_descriptors(sift, rois)  

    return rois

def load_image(filepath, show=False):
    img = cv2.imread(filepath, 0)
    if show:
        cv2.imshow(filepath, img)
        ch = cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img

def get_iris_boundaries(img, show=False):
    # Finding iris inner boundary
    pupil_circle = find_pupil(img)

    if pupil_circle is None:
        print('ERROR: Pupil circle not found!')
        return None, None

    # Finding iris outer boundary
    radius_range = int(math.ceil(pupil_circle[2] * 1.5))
    multiplier = 0.25
    center_range = int(math.ceil(pupil_circle[2] * multiplier)) 
    ext_iris_circle = find_ext_iris(
                        img, pupil_circle, center_range, radius_range)

    while ext_iris_circle is None and multiplier <= 0.7:
        multiplier += 0.05
        print('Searching exterior iris circle with multiplier ' + str(multiplier))
        center_range = int(math.ceil(pupil_circle[2] * multiplier))
        ext_iris_circle = find_ext_iris(img, pupil_circle, center_range, radius_range)
    if ext_iris_circle is None:
        print('ERROR: Exterior iris circle not found!')
        return None, None
    
    if show:
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        draw_circles(cimg, pupil_circle, ext_iris_circle, center_range, radius_range)
        cv2.imshow('iris boundaries', cimg)
        ch = cv2.waitKey(0)
        cv2.destroyAllWindows()

    return pupil_circle, ext_iris_circle

def find_pupil(img):
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                                param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

        return circles[0, 0]

    return None

def get_equalized_iris(img, ext_iris_circle, pupil_circle, show=False):
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_mask = np.zeros_like(img)

    cv2.circle(img_mask, (pupil_circle[0], pupil_circle[1]), pupil_circle[2], 255, -1)
    cv2.circle(img_mask, (ext_iris_circle[0], ext_iris_circle[1]), ext_iris_circle[2], 255, -1)

    masked_img = cv2.bitwise_and(img, img_mask)
    hist_img = cv2.equalizeHist(masked_img)

    if show:
        cv2.imshow('Equalized iris', hist_img)
        ch = cv2.waitKey(0)
        cv2.destroyAllWindows()

    return hist_img

def get_rois(img, pupil_circle, ext_iris_circle, show=False):
    inner_radius = int(pupil_circle[2] * 1.75)
    outer_radius = int(ext_iris_circle[2] * 0.85)

    inner_circles = []
    outer_circles = []

    angle_increment = 2 * math.pi / 24  # dividing the iris region into 24 sectors

    for i in range(0, 24):
        angle = i * angle_increment

        inner_circles.append((
            int(pupil_circle[0] + inner_radius * math.cos(angle)),
            int(pupil_circle[1] + inner_radius * math.sin(angle)),
            int(0.15 * pupil_circle[2])
        ))

        outer_circles.append((
            int(ext_iris_circle[0] + outer_radius * math.cos(angle)),
            int(ext_iris_circle[1] + outer_radius * math.sin(angle)),
            int(0.07 * ext_iris_circle[2])
        ))

    rois = []

    for i in range(len(inner_circles)):
        inner_roi = get_roi(img, inner_circles[i])
        outer_roi = get_roi(img, outer_circles[i])

        rois.append({
            'inner_circle': inner_circles[i],
            'outer_circle': outer_circles[i],
            'inner_roi': inner_roi,
            'outer_roi': outer_roi
        })

    if show:
        for roi in rois:
            cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            draw_circles(cimg, roi['inner_circle'], roi['outer_circle'], show=False)
            cv2.imshow('ROI', cimg)
            ch = cv2.waitKey(0)
            cv2.destroyAllWindows()

    return rois

def get_roi(img, circle):
    mask = np.zeros_like(img)
    cv2.circle(mask, (circle[0], circle[1]), circle[2], 255, -1)

    roi = cv2.bitwise_and(img, mask)

    return roi

def draw_circles(img, inner_circle, outer_circle, center_range=None, radius_range=None, show=True):
    if center_range is not None and radius_range is not None:
        cv2.circle(img, (inner_circle[0], inner_circle[1]), inner_circle[2], (0, 255, 0), 2)
        cv2.circle(img, (outer_circle[0], outer_circle[1]), outer_circle[2], (0, 255, 0), 2)
        cv2.circle(img, (inner_circle[0], inner_circle[1]), center_range, (0, 0, 255), 2)
        cv2.circle(img, (outer_circle[0], outer_circle[1]), radius_range, (0, 0, 255), 2)
    else:
        cv2.circle(img, (inner_circle[0], inner_circle[1]), inner_circle[2], (0, 255, 0), 2)
        cv2.circle(img, (outer_circle[0], outer_circle[1]), outer_circle[2], (0, 255, 0), 2)

    if show:
        cv2.imshow('Circles', img)
        ch = cv2.waitKey(0)
        cv2.destroyAllWindows()

def load_keypoints(sift, rois, show=False):
    for roi in rois:
        _, inner_keypoints = sift.detectAndCompute(roi['inner_roi'], None)
        roi['inner_keypoints'] = inner_keypoints

        _, outer_keypoints = sift.detectAndCompute(roi['outer_roi'], None)
        roi['outer_keypoints'] = outer_keypoints

        if show:
            img_inner = cv2.drawKeypoints(
                roi['inner_roi'], roi['inner_keypoints'], None, color=(0, 255, 0), flags=0)
            img_outer = cv2.drawKeypoints(
                roi['outer_roi'], roi['outer_keypoints'], None, color=(0, 255, 0), flags=0)

            cv2.imshow('Inner Keypoints', img_inner)
            cv2.imshow('Outer Keypoints', img_outer)
            ch = cv2.waitKey(0)
            cv2.destroyAllWindows()

def load_descriptors(sift, rois):
    for roi in rois:
        roi['inner_descriptors'] = sift.compute(roi['inner_roi'], roi['inner_keypoints'])[1]
        roi['outer_descriptors'] = sift.compute(roi['outer_roi'], roi['outer_keypoints'])[1]

def find_ext_iris(img, pupil_circle, center_range, radius_range):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 100,
                                param1=50, param2=30, minRadius=radius_range - 3, maxRadius=radius_range + 3)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            center = (i[0], i[1])

            if is_valid_center(center, pupil_circle, center_range):
                return i

    return None

def is_valid_center(center, pupil_circle, center_range):
    distance = math.sqrt((center[0] - pupil_circle[0]) ** 2 + (center[1] - pupil_circle[1]) ** 2)
    return distance >= center_range - 3 and distance <= center_range + 3

def get_all_matches(rois1, rois2, ratio, num_matches, threshold, show=False):
    all_matches = []
    for i, roi1 in enumerate(rois1):
        best_matches = []
        best_match_index = -1
        for j, roi2 in enumerate(rois2):
            matches = get_matches(roi1['inner_descriptors'], roi2['inner_descriptors'], ratio)
            if len(matches) > len(best_matches):
                best_matches = matches
                best_match_index = j
        if best_match_index != -1:
            all_matches.append((i, best_match_index, len(best_matches)))

    sorted_matches = sorted(all_matches, key=lambda x: x[2], reverse=True)
    final_matches = []
    for match in sorted_matches:
        if match[2] >= threshold:
            final_matches.append(match)

    if show:
        img1 = cv2.cvtColor(rois1[0]['inner_roi'], cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(rois2[0]['inner_roi'], cv2.COLOR_GRAY2BGR)
        img3 = cv2.drawMatchesKnn(
            rois1[0]['inner_roi'], rois1[0]['inner_keypoints'],
            rois2[0]['inner_roi'], rois2[0]['inner_keypoints'],
            [match[0] for match in final_matches], None, flags=2)

        plt.imshow(img3)
        plt.show()

    return final_matches

def get_matches(descriptors1, descriptors2, ratio):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return good_matches


# Example usage
compare_images(r'C:\Users\DELL\Iris-recognition\Dataset\Joe\right\chingycr1.jpg', r'C:\Users\DELL\Iris-recognition\Dataset\Angela\left\fatmal1.jpg')
compare_binfiles("file1.bin", "file2.bin")
