import cv2 as cv
import numpy as np
from math import sqrt

# cod (intermediat, nefinisat) pentru a extrage template urile folosite in codul propriu-zis
# codul din acest fisier are rol demonstrativ, nu trebuie neaparat rulat
# in plus, path-urile nu sunt configurate

def l2(x0, y0, x1, y1):
    return (y1 - y0) ** 2 + (x1 - x0) ** 2

def show_image(image, title = "img"):

    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def show_images(images):

    for i in range(len(images)):
        cv.imshow(f"title{i}", images[i])

    cv.waitKey(0)
    cv.destroyAllWindows()

def get_img(img_i, jigsaw = False):

    imgname = img_i
    if img_i < 10:
        imgname = f"0{img_i}"

    if jigsaw is False:

        img = cv.imread(f"antrenare/clasic/{imgname}.jpg")
        img = cv.resize(img, (0, 0), fx = 0.2, fy = 0.2)

        return img

    else:

        img = cv.imread(f"antrenare/jigsaw/{imgname}.jpg")
        img = cv.resize(img, (0, 0), fx = 0.2, fy = 0.2)

        return img

def find_corners(image, show = False):

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    image_m_blur = cv.medianBlur(image, 3)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5) 
    image_sharpened = cv.addWeighted(image_m_blur, 1.25, image_g_blur, -0.7, 0)
    _, thresh = cv.threshold(image_sharpened, 25, 255, cv.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.erode(thresh, kernel)
    
    if show:
        show_image(image_m_blur, "median blur")
        show_image(image_g_blur, "gaussian blurred")
        show_image(image_sharpened, "sharpened")    
        show_image(thresh, "threshold of blur")

    edges =  cv.Canny(thresh, 150, 400)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
   
    for i in range(len(contours)):
        if(len(contours[i]) > 3):

            possible_top_left = None
            possible_bottom_right = None

            for point in contours[i].squeeze():

                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1] :
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis = 1)

            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]

            if cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right], [possible_bottom_left]])) > max_area:
                
                max_area = cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right], [possible_bottom_left]]))
                
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left
    
    image_copy = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)

    cv.circle(image_copy, tuple(top_left), 4, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(top_right), 4, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(bottom_left), 4, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(bottom_right), 4, (0, 0, 255), -1)

    if show:
        show_image(image_copy, "detected corners")
    
    return top_left, top_right, bottom_left, bottom_right

def get_center(img_i, show = False, jigsaw = False):

    img = get_img(img_i, jigsaw)
    tl, tr, _, _ = find_corners(img, show)

    if show:
        show_image(img)

    l = round(sqrt(l2(*tl, *tr)))

    slope = (tr[1] - tl[1]) / (tr[0] - tl[0])
    angle = np.degrees(np.arctan(slope))
    
    rotation_matrix = cv.getRotationMatrix2D((int(tl[0]), int(tl[1])), angle, 1)
    img = cv.warpAffine(img, rotation_matrix, (int(img.shape[1]), int(img.shape[0])))

    if show:
        show_image(img)

    img = img[tl[1]: tl[1] + l, tl[0]: tl[0] + l, :]

    if show:
        show_image(img)

    return img

def _get_border_samples():

    img = get_center(1, show = False, jigsaw = True)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    l = img.shape[0]
    chunk_l = l // 9

    def _check_line(x1, y1, x2, y2):

        d0 = abs(x1 - x2)
        d1 = abs(y1 - y2)

        if (d0 > chunk_l // 5) and (d1 > chunk_l // 5):
            return False

        return True

    #show_image(img)

    image_m_blur = cv.medianBlur(img, 7)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 11) 
    image_sharpened = cv.addWeighted(image_m_blur, 1.5, image_g_blur, -1, 0)
    _, thresh = cv.threshold(image_sharpened, 10, 255, cv.THRESH_BINARY)

    #show_image(thresh)
    edges =  cv.Canny(thresh, 150, 400)
    lines = cv.HoughLinesP(image = edges, rho = 1, theta = np.pi / 180, threshold = 25, minLineLength = 0, maxLineGap = chunk_l // 3)

    edges = np.zeros(edges.shape)

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if _check_line(x1, y1, x2, y2):
            cv.line(edges, (x1, y1), (x2, y2), (255, 255, 255), 3)

    #show_image(edges)
    edges = edges.astype(np.uint8)

    upedge = edges[int(4.1 * chunk_l): int(4.8 * chunk_l), int(6.7 * chunk_l): int(7.3 * chunk_l)]
    lowedge = edges[int(0.7 * chunk_l): int(1.3 * chunk_l), int(1.1 * chunk_l): int(1.8 * chunk_l)]

    #upedge = cv.GaussianBlur(upedge, (0, 0), 3)
    #lowedge = cv.GaussianBlur(lowedge, (0, 0), 3)

    upedge *= 2
    lowedge *= 2

    cv.imwrite("up_edge.jpg", upedge)
    cv.imwrite("low_edge.jpg", lowedge)

def _extract_digits():

    def _jigsaw_gray():

        for img_i in [14, 37, 4, 15]:

            img = get_center(img_i, show = False, jigsaw = True)
            #show_image(img)

            l = img.shape[0]
            chunk_l = l // 9

            for i in range(9):
                for j in range(9):
            
                    chunk = img[i * chunk_l: (i + 1) * chunk_l, 
                                j * chunk_l: (j + 1) * chunk_l, :]
                    chunk = cv.cvtColor(chunk, cv.COLOR_RGB2GRAY)
                    _, chunk = cv.threshold(chunk, 130, 255, cv.THRESH_BINARY)

                    chunk = cv.medianBlur(chunk, 3)
                    chunk = chunk[chunk_l // 5: chunk_l * 4 // 5, chunk_l // 5: chunk_l * 4 // 5]
                    
                    if img_i == 4:

                        if i == 1 and j == 1:
                            show_image(chunk)
                            cv.imwrite("digit_1_jgray.jpg", chunk)

                        if i == 1 and j == 0:
                            show_image(chunk)
                            cv.imwrite("digit_9_jgray.jpg", chunk)

                        if i == 8 and j == 0:
                            show_image(chunk)
                            cv.imwrite("digit_6_jgray.jpg", chunk)

                    elif img_i == 15:

                        if i == 1 and j == 3:
                            show_image(chunk)
                            cv.imwrite("digit_4_jgray.jpg", chunk)

                        if i == 1 and j == 0:
                            show_image(chunk)
                            cv.imwrite("digit_5_jgray.jpg", chunk)

                        if i == 1 and j == 4:
                            show_image(chunk)
                            cv.imwrite("digit_7_jgray.jpg", chunk)

                        if i == 1 and j == 2:
                            show_image(chunk)
                            cv.imwrite("digit_8_jgray.jpg", chunk)

                    elif img_i == 37:

                        if i == 1 and j == 3:
                            show_image(chunk)
                            cv.imwrite("digit_2_jgray.jpg", chunk)

                    elif img_i == 14:

                        if i == 7 and j == 2:
                            show_image(chunk)
                            cv.imwrite("digit_3_jgray.jpg", chunk)

    def _jigsaw_color():

        for img_i in [23, 32, 40, 31]:

            img = get_center(img_i, show = False, jigsaw = True)
            #show_image(img)
            
            l = img.shape[0]
            chunk_l = l // 9

            for i in range(9):
                for j in range(9):
            
                    chunk = img[i * chunk_l: (i + 1) * chunk_l, 
                                j * chunk_l: (j + 1) * chunk_l, :]
                    chunk = cv.cvtColor(chunk, cv.COLOR_RGB2GRAY)
                    _, chunk = cv.threshold(chunk, 130, 255, cv.THRESH_BINARY)

                    chunk = cv.medianBlur(chunk, 3)
                    chunk = chunk[chunk_l // 4: chunk_l * 3 // 4, chunk_l // 4: chunk_l * 3 // 4]
                    
                    if img_i == 23:

                        if i == 0 and j == 1:
                            show_image(chunk)
                            cv.imwrite("digit_8_jcolor.jpg", chunk)

                        if i == 0 and j == 6:
                            show_image(chunk)
                            cv.imwrite("digit_7_jcolor.jpg", chunk)

                    elif img_i == 32:

                        if i == 0 and j == 1:
                            show_image(chunk)
                            cv.imwrite("digit_6_jcolor.jpg", chunk)

                        if i == 0 and j == 4:
                            show_image(chunk)
                            cv.imwrite("digit_1_jcolor.jpg", chunk)

                    elif img_i == 40:

                        if i == 0 and j == 3:
                            show_image(chunk)
                            cv.imwrite("digit_3_jcolor.jpg", chunk)

                        if i == 0 and j == 4:
                            show_image(chunk)
                            cv.imwrite("digit_2_jcolor.jpg", chunk)

                        if i == 0 and j == 6:
                            show_image(chunk)
                            cv.imwrite("digit_5_jcolor.jpg", chunk)

                        if i == 0 and j == 7:
                            show_image(chunk)
                            cv.imwrite("digit_9_jcolor.jpg", chunk)

                    elif img_i == 31:

                        if i == 0 and j == 6:
                            show_image(chunk)
                            cv.imwrite("digit_4_jcolor.jpg", chunk)

    def _clasic():

        for img_i in [1, 3]:

            img = get_center(img_i, show = False, jigsaw = False)

            l = img.shape[0]
            chunk_l = l // 9

            for i in range(9):
                for j in range(9):
            
                    chunk = img[i * chunk_l: (i + 1) * chunk_l, 
                                j * chunk_l: (j + 1) * chunk_l, :]
                    chunk = cv.cvtColor(chunk, cv.COLOR_RGB2GRAY)
                    _, chunk = cv.threshold(chunk, 130, 255, cv.THRESH_BINARY)

                    chunk = cv.medianBlur(chunk, 3)
                    chunk = chunk[chunk_l // 4: chunk_l * 3 // 4, chunk_l // 4: chunk_l * 3 // 4]
                    
                    if img_i == 1:

                        if i == 0 and j == 6:
                            show_image(chunk)
                            cv.imwrite("digit_5_clasic.jpg", chunk)

                        if i == 0 and j == 1:
                            show_image(chunk)
                            cv.imwrite("digit_6_clasic.jpg", chunk)

                        if i == 0 and j == 2:
                            show_image(chunk)
                            cv.imwrite("digit_8_clasic.jpg", chunk)

                        if i == 1 and j == 2:
                            show_image(chunk)
                            cv.imwrite("digit_4_clasic.jpg", chunk)

                    elif  img_i == 3:

                        if i == 0 and j == 2:
                            show_image(chunk)
                            cv.imwrite("digit_1_clasic.jpg", chunk)

                        if i == 0 and j == 5:
                            show_image(chunk)
                            cv.imwrite("digit_2_clasic.jpg", chunk)

                        if i == 0 and j == 7:
                            show_image(chunk)
                            cv.imwrite("digit_3_clasic.jpg", chunk)

                        if i == 0 and j == 6:
                            show_image(chunk)
                            cv.imwrite("digit_7_clasic.jpg", chunk)

                        if i == 0 and j == 1:
                            show_image(chunk)
                            cv.imwrite("digit_9_clasic.jpg", chunk)

    _clasic()
    _jigsaw_gray()
    _jigsaw_color()

if __name__ == "__main__":

    _get_border_samples()
    _extract_digits()
