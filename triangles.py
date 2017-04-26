import cv2 as cv
import numpy as np
import math
import itertools
from matplotlib import pyplot as plt
import time
# import matplotlib

np.set_printoptions(suppress=True)


def show(image, title="window", keep_window=False):
    """
    shows the image with some optional modifiers
    saves writing three statements
    """
    if image.dtype != "uint8":
        # the array is int32, changing to uint8
        new_img = image.astype("uint8")
    else:
        new_img = image
    cv.imshow("cv - " + title, new_img)
    cv.waitKey(0)
    if not keep_window:
        cv.destroyAllWindows()


def sort_array(unsorted_array, column=1):
    """
    sorts array by the column
    """
    return unsorted_array[unsorted_array[:, column].argsort()]


def polar_to_cartesian(rho, theta):
    """
    changes polar coordinates of a line to cartesian
    """
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)
    return x1, y1, x2, y2


def get_line_coeff(rho, theta):
    """
    calculates k and n for slope intercept form of a line from polar coordinates
    """
    # print("rho is {} and theta is {}".format(rho, theta))
    try:
        k = -math.cos(theta)/math.sin(theta)
        n = rho/math.sin(theta)
    except ZeroDivisionError:
        # this means the line is vertical so k and n don't exist
        k = math.inf
        n = rho
    # print("k is {} n is {}".format(k, n))
    return k, n


def get_line_intersection(line1, line2):
    """
    returns x and y of two lines intersecting, and None if they don't intersect
    """
    # print("line1 is {} line2 is {}".format(line1, line2))
    rho1, theta1 = line1
    rho2, theta2 = line2
    k1, n1 = get_line_coeff(*line1)
    k2, n2 = get_line_coeff(*line2)

    if k1 == k2:
        # print("Both lines have the same slope, there are no intersections")
        return
    elif math.inf in (k1, n1, k2, n2):
        if k1 is math.inf:
            x = line1[0]
            y = k2 * x + n2
        elif k2 is math.inf:
            x = line2[0]
            y = k1 * x + n1
    else:
        x = (n2 - n1) / (k1 - k2)
        y = (k1*n2 - k2*n1) / (k1 - k2)
    # print("x is {} y is {}".format(x, y))
    return math.floor(x), math.floor(y)


def is_same(line_1, line_2):
    """
    returns True if the lines are almost the same
    used to cancel duplicated lines from noise
    lines are considered the same if:
        the angle between them is less than 12 degrees and
        the distance between them is less than 20 pixels at the base
    """
    rho_margin = 20  # arbitrary number, in pixels
    theta_margin = math.pi / 30  # arbitrary number
    rho_1, theta_1 = line_1.flat
    rho_2, theta_2 = line_2.flat
    rho_difference = abs(rho_1 - rho_2)
    theta_difference = abs(theta_1 - theta_2)
    if (theta_1 >= (math.pi - theta_margin)) \
            ^ (theta_2 >= (math.pi - theta_margin)):
        rho_difference = abs(rho_1 + rho_2)
        theta_difference = abs(theta_1 - theta_2 - math.pi)
    if rho_difference > rho_margin:
        return False
    if theta_difference > theta_margin:
        return False
    # print("found same {} {}".format(line_1, line_2))
    return True


def get_unique_lines(lines):
    """
    returns unique lines in a list
    """
    unique_lines = [lines[0]]
    lines = lines[1:]
    for line in lines:
        for u_line in unique_lines:
            if is_same(line, u_line):
                break
        else:
            unique_lines.append(line)
    return np.array(unique_lines)


triangles = cv.imread("triangles.png")
tri_grayscale = cv.cvtColor(triangles, cv.COLOR_RGB2GRAY)
tri_smooth = cv.blur(tri_grayscale, (2, 2))
thresh, tri_bw = cv.threshold(
    tri_grayscale, 60, 255, cv.THRESH_BINARY)
tri_bw = cv.morphologyEx(tri_bw, cv.MORPH_OPEN, (3, 3))
N, labels, stats, centroids = cv.connectedComponentsWithStats(tri_bw)
tri_edges = cv.Canny(tri_smooth, 100, 200)
lines = cv.HoughLines(tri_edges, 0.7, math.pi / 180, 40, 0)
triangles_backup = np.copy(triangles)
results = np.zeros(triangles.shape)


number_of_triangles = 0
solution = np.zeros(tri_grayscale.shape)

# runs over each component in order to reduce false matches
for i in range(1, N):
    triangles = np.copy(triangles_backup)
    component = np.asarray((labels == i) * 255, "uint8")
    component_edges = cv.Canny(component, 100, 200)
    lines = cv.HoughLines(component_edges, 1, math.pi / 180, 30, 0)
    if lines is None:
        continue
    lines = lines.squeeze()
    unique_lines = get_unique_lines(lines)
    for line in unique_lines:  # draws lines
        # print("line is {}".format(line))
        x1, y1, x2, y2 = polar_to_cartesian(line[0], line[1])
        cv.line(triangles, (x1, y1), (x2, y2), 255, 1)
    # print("filtered lines shape {}".format(unique_lines.shape))
    # draws a ruler
    for i in range(0, triangles.shape[1], 10):
        triangles[0:10, i] = (255, 255, 255)
        if i % 50 == 0:
            triangles[11:20, i] = (255, 255, 255)
            cv.putText(triangles, str(i), (i + 2, 20), cv.FONT_HERSHEY_PLAIN,
                       0.7, (255, 255, 255))
    n_intersections = 0
    intersections_list = []
    # combines lines into pairs to check for intersections
    for line1, line2 in itertools.combinations(unique_lines,2):
        # checks whether lines have intersections
        intersection = get_line_intersection(line1, line2)
        if intersection is not None \
                and (0 <= intersection[0] <= triangles.shape[1]) \
                and (0 <= intersection[1] <= triangles.shape[0]):
            intersections_list.append(intersection)
            n_intersections += 1
            # print("intersection in {}".format(intersection))
            cv.line(triangles, intersection, intersection, (0, 255, 0), 3)
    # three lines and three intersection, seems we have a triangle
    if n_intersections == 3 and len(unique_lines) == 3:
        lines = []
        for point_1, point_2 in itertools.combinations(intersections_list, 2):
            line_length = cv.norm(point_1, point_2)
            lines.append(line_length)
        s = sum(lines)/2
        # print("s is {} a is {} b is {} c is {}".format(s, lines[0],
        #                                                lines[1], lines[2]))
        area_of_triangle = math.sqrt(s*(s-lines[0])*(s-lines[1])*(s-lines[2]))
        area_of_component = np.sum(component > 0)
        # if the calculated triangle area and image component area differ by
        # more than 5 percent it's not a good match
        if (area_of_triangle - area_of_component) > (0.05 * area_of_component):
            continue
        number_of_triangles += 1
        solution += component
cv.putText(solution, "I found {} triangles".format(number_of_triangles),
           (160, 200), cv.FONT_HERSHEY_PLAIN,
           2, (255, 255, 255))
show(solution, title="Here are the triangles I found")
print("The number of triangles is {}.".format(number_of_triangles))
