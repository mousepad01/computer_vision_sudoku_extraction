import cv2 as cv
import numpy as np
from math import sqrt

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

def get_img(img_i):

    imgname = img_i
    if img_i < 10:
        imgname = f"0{img_i}"

    img = cv.imread(f"antrenare/clasic/{imgname}.jpg")
    img = cv.resize(img, (0, 0), fx = 0.2, fy = 0.2)

    return img

def save_mat_s1(img_i, m, digits = False):

    solname = img_i
    if img_i < 10:
        solname = f"0{img_i}"

    if digits:
        solname = f"myres/{solname}_bonus_gt.txt"
    else:
        solname = f"myres/{solname}_gt.txt"

    fsol = open(solname, "w+")
    for i in range(9):

        for j in range(9):
            fsol.write(m[i][j])

        fsol.write("\n")

    fsol.close()

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

def get_center(img_i, show = False):

    img = get_img(img_i)
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

def solve_task_1(img_i, identify_digits = False, show = False):

    img = get_center(img_i, show)

    l = img.shape[0]
    chunk_l = l // 9

    empty = np.full((int(chunk_l * 0.7), int(chunk_l * 0.7)), np.uint8(255))
        
    def _is_empty(ch):

        # checks whether there is a contiguous "only-white" portion in the iamge
        # of (3/4, 2/4) of the chunk size
        # with a small error margin determined by grid search

        TOLERATED_ERR = 0.003

        match = cv.matchTemplate(ch, empty, cv.TM_SQDIFF_NORMED)
        minval, _, _, _ = cv.minMaxLoc(match)

        if minval < TOLERATED_ERR:
            return True

        return False

    def _with_digits(): 

        res = [["o" for _ in range(9)] for _ in range(9)]

        def _get_co(ch, template):

            match = cv.matchTemplate(ch, template, cv.TM_CCOEFF_NORMED)
            _, maxval, _, _ = cv.minMaxLoc(match)

            return maxval
        
        digits = [np.zeros((chunk_l // 2, chunk_l // 2), np.uint8)]
        for i in range(1, 10):

            d = cv.imread(f"digit_{i}.jpg")
            d = cv.resize(d, (chunk_l // 2, chunk_l // 2))
            d = cv.cvtColor(d, cv.COLOR_BGR2GRAY)

            digits.append(d)

        for i in range(9):
            for j in range(9):

                off = i

                chunk = img[i * chunk_l + off: (i + 1) * chunk_l + off, 
                            j * chunk_l: (j + 1) * chunk_l, :]
                chunk = cv.cvtColor(chunk, cv.COLOR_RGB2GRAY)
                _, chunk = cv.threshold(chunk, 120, 255, cv.THRESH_BINARY)

                chunk = cv.medianBlur(chunk, 3)

                if _is_empty(chunk):
                    continue

                res[i][j] = chr(np.argmax([-1000000] + [_get_co(chunk, d) for d in digits[1:]]) + ord('0'))

        return res

    def _extract_digits():

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
                        cv.imwrite("digit_5.jpg", chunk)

                    if i == 0 and j == 1:
                        show_image(chunk)
                        cv.imwrite("digit_6.jpg", chunk)

                    if i == 0 and j == 2:
                        show_image(chunk)
                        cv.imwrite("digit_8.jpg", chunk)

                    if i == 1 and j == 2:
                        show_image(chunk)
                        cv.imwrite("digit_4.jpg", chunk)

                elif  img_i == 3:

                    if i == 0 and j == 2:
                        show_image(chunk)
                        cv.imwrite("digit_1.jpg", chunk)

                    if i == 0 and j == 5:
                        show_image(chunk)
                        cv.imwrite("digit_2.jpg", chunk)

                    if i == 0 and j == 7:
                        show_image(chunk)
                        cv.imwrite("digit_3.jpg", chunk)

                    if i == 0 and j == 6:
                        show_image(chunk)
                        cv.imwrite("digit_7.jpg", chunk)

                    if i == 0 and j == 1:
                        show_image(chunk)
                        cv.imwrite("digit_9.jpg", chunk)

    def _without_digits():
        
        res = [["o" for _ in range(9)] for _ in range(9)]

        for i in range(9):
            for j in range(9):

                off = i

                chunk = img[i * chunk_l + off: (i + 1) * chunk_l + off, 
                            j * chunk_l: (j + 1) * chunk_l, :]
                chunk = cv.cvtColor(chunk, cv.COLOR_RGB2GRAY)
                _, chunk = cv.threshold(chunk, 120, 255, cv.THRESH_BINARY)

                chunk = cv.medianBlur(chunk, 3)

                if _is_empty(chunk):
                    continue

                res[i][j] = "x"

        return res

    if identify_digits:
        return _with_digits()
    else:
        return _without_digits()

def check_task_1():

    def _cmp_m(fst, snd):

        for i in range(9):
            for j in range(9):
                
                if snd[i][j] != fst[i][j]:
                    return False

        return True

    ok = 0
    ok_digit = 0

    IMG_CNT = 20
    for i in range(1, IMG_CNT + 1):

        solname_ = i
        if i < 10:
            solname_ = f"0{i}"

        solname_d = f"{solname_}_bonus_gt.txt"
        solname = f"{solname_}_gt.txt"

        f_res = open(f"myres/{solname}")
        f_org = open(f"antrenare/clasic/{solname}")
        f_res_d = open(f"myres/{solname_d}")
        f_org_d = open(f"antrenare/clasic/{solname_d}")

        res = f_res.read().split()
        org = f_org.read().split()
        res_d = f_res_d.read().split()
        org_d = f_org_d.read().split()

        f_res.close()
        f_org.close()
        f_res_d.close()
        f_org_d.close()

        if _cmp_m(res, org) is True:
            ok += 1

        if _cmp_m(res_d, org_d) is True:
            ok_digit += 1

    print(f"recunoastere fara cifre: {ok} / 20")
    print(f"recunoastere cu cifre: {ok_digit} / 20")

    return ok, ok_digit

if __name__ == "__main__":
    
    IMG_CNT = 20
    for i in range(1, IMG_CNT + 1):

        m = solve_task_1(i, False, False)
        m_d = solve_task_1(i, True, False)

        save_mat_s1(i, m, False)
        save_mat_s1(i, m_d, True)

    check_task_1()
