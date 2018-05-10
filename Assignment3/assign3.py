import cv2
import numpy as np


file = open('InvIntrinsicIR', 'r')
invIntrinsicIR = []
for line in file:
    invIntrinsicIR.append(line.strip().split(','))
invIntrinsicIR = np.array(invIntrinsicIR).astype(np.float)

file = open('IntrinsicRGB', 'r')
intrinsicRGB = []
for line in file:
    intrinsicRGB.append(line.strip().split(','))
intrinsicRGB = np.array(intrinsicRGB).astype(np.float)

file = open('TransformationD-C', 'r')
transformationD_C = []
for line in file:
    transformationD_C.append(line.strip().split(','))
transformationD_C = np.array(transformationD_C).astype(np.float)
file.close()

def generateColorizedImage(color_image, depth_image):
    # load the initial RGB image
    color = cv2.imread(color_image)
    colorHeight, colorWidth, channel = color.shape
    # load the initial depth image
    depth = cv2.imread(depth_image, cv2.IMREAD_ANYDEPTH)
    depthHeight, depthWidth = depth.shape
    blank_image = np.zeros((depthHeight, depthWidth, 3), np.uint8)
    # generate colorize image
    for i in range(depthHeight):
        for j in range(depthWidth):
            matrix = np.matmul(np.array([j, i, 1]), invIntrinsicIR)
            if depth[i][j] > 0:
                matrix = np.multiply(matrix, depth[i][j])
                matrix = np.append(matrix, 1)
                matrix = np.matmul(matrix, transformationD_C)
                a = matrix[2]
                matrix = matrix / a
                matrix = matrix[0:3]
                matrix = np.matmul(matrix, intrinsicRGB)
                x_pix = int(matrix[0])
                y_pix = int(matrix[1])
                if(x_pix > 0 and x_pix < colorWidth and y_pix > 0 and y_pix < colorHeight):
                    blank_image[i][j] = color[y_pix][x_pix]
    return blank_image

def color_detect(image, a, b):
    lower = np.array([0,60,40])
    upper = np.array([16,255,255])
    img = cv2.imread(image)
    image = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    # kernel = np.array([7, 7])
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.erode(mask, None, iterations = 2)
    mask = cv2.dilate(mask, None, iterations = 1)
    cv2.imwrite(b, mask)
    # cv2.imshow(b, mask)
    cnt = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    c1 = max(cnt, key = cv2.contourArea)
    cnt.remove(c1)
    c2 = max(cnt, key = cv2.contourArea)
    ((x1, y1), radius1) = cv2.minEnclosingCircle(c1)
    ((x2, y2), radius2) = cv2.minEnclosingCircle(c2)
    cv2.rectangle(img, (int(x1 - radius1), int(y1 - radius1)),(int(x1 + radius1), int(y1 + radius1)),(0, 255, 0), 2)
    cv2.rectangle(img, (int(x2 - radius2), int(y2 - radius2)),(int(x2 + radius2), int(y2 + radius2)),(0, 255, 0), 2)
    #cv2.circle(img, (int(x2), int(y2)), int(radius2),(0, 255, 0), 2)
    cv2.imwrite(a, img)
    cv2.imshow(a, img)
    if (cv2.waitKey(0) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
    return (x1, y1), (x2, y2)


def compute_velocity(image1, image2, a1, b1, a2, b2):
    depth1 = cv2.imread(image1, cv2.IMREAD_ANYDEPTH)
    depth2 = cv2.imread(image2, cv2.IMREAD_ANYDEPTH)
    (x1, y1) = (int(b1), int(a1))
    (x2, y2) = (int(b2), int(a2))
    matrix1 = np.matmul(np.array([x1, y1, 1]), invIntrinsicIR)
    # matrix1[2] = np.multiply(matrix1[2], depth1[x1][y1])
    matrix1 = np.multiply(matrix1, depth1[x1][y1])
    matrix2 = np.matmul(np.array([x2, y2, 1]), invIntrinsicIR)
    # matrix2[2] = np.multiply(matrix2[2], depth2[x2][y2])
    matrix2 = np.multiply(matrix2, depth1[x1][y1])
    distance = np.sqrt(np.square(matrix1[1] - matrix2[1]) + np.square(matrix1[2] - matrix2[2]))
    # print(distance)
    # print(matrix1[0])
    # print(matrix2[0])
    velocity = 1.0 * distance / 1.3
    print(velocity)
    return velocity


colorize_image1 = generateColorizedImage('color-63647317626781.png', 'depth-63647317626781.png' )
colorize_image2 = generateColorizedImage('color-63647317628081.png', 'depth-63647317628081.png' )
cv2.imwrite("colorize_image1.jpg", colorize_image1)
cv2.imwrite("colorize_image2.jpg", colorize_image2)
(initial_x1, initial_y1), (initial_x2, initial_y2) = color_detect("colorize_image1.jpg", "initial_detect_1.jpg", "tmp_1.jpg")
(final_x1, final_y1), (final_x2, final_y2) = color_detect("colorize_image2.jpg", "final_detect_1.jpg", "tmp_2.jpg")
velocity1 = compute_velocity('depth-63647317626781.png', 'depth-63647317628081.png', initial_x1, initial_y1, final_x1, final_y1)
velocity2 = compute_velocity('depth-63647317626781.png', 'depth-63647317628081.png', initial_x2, initial_y2, final_x2, final_y2)
print('relative velocity =', velocity1 + velocity2, "mm/s")
