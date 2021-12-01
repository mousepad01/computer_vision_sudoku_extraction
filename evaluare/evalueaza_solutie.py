import cv2 as cv
import numpy as np
from math import sqrt

# Stanciu Andrei Calin

# toate path-urile folosite (format linux)

DATA_PATH_CLASIC = "../test/clasic/"
DATA_PATH_JIGSAW = "../test/jigsaw/"
PREDICTION_PATH_CLASIC = "../fisiere_solutie/Stanciu_Calin_331/clasic/"
PREDICTION_PATH_JIGSAW = "../fisiere_solutie/Stanciu_Calin_331/jigsaw/"
TEMPLATE_PATH = "templates/"

# --------------

CLASIC_IMG_CNT = 20
JIGSAW_IMG_CNT = 40

# common code for both task_1 and task_2

def l2(x0, y0, x1, y1):
    return (y1 - y0) ** 2 + (x1 - x0) ** 2

def get_img(img_i, jigsaw = False):

    imgname = img_i
    if img_i < 10:
        imgname = f"0{img_i}"

    if jigsaw is False:

        img = cv.imread(f"{DATA_PATH_CLASIC}{imgname}.jpg")
        img = cv.resize(img, (0, 0), fx = 0.2, fy = 0.2)

        return img

    else:

        img = cv.imread(f"{DATA_PATH_JIGSAW}{imgname}.jpg")
        img = cv.resize(img, (0, 0), fx = 0.2, fy = 0.2)

        return img

def save_solution(img_i, m, digits = False, jigsaw = False):

    if digits:
        solname = f"{img_i}_bonus_predicted.txt"
    else:
        solname = f"{img_i}_predicted.txt"

    if jigsaw:
        solname = f"{PREDICTION_PATH_JIGSAW}{solname}"
    else:
        solname = f"{PREDICTION_PATH_CLASIC}{solname}"

    fsol = open(solname, "w+")
    for i in range(9):

        for j in range(9):
            fsol.write(m[i][j])

        if i != 8:
            fsol.write("\n")

    fsol.close()

def find_corners(image):

    # cod preluat in mare parte din laborator - doar unii parametri au fost modificati

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    image_m_blur = cv.medianBlur(image, 3)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5) 
    image_sharpened = cv.addWeighted(image_m_blur, 1.25, image_g_blur, -0.7, 0)
    _, thresh = cv.threshold(image_sharpened, 25, 255, cv.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.erode(thresh, kernel)

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

    return top_left, top_right, bottom_left, bottom_right

def get_center(img_i, jigsaw = False):

    img = get_img(img_i, jigsaw)
    tl, tr, _, _ = find_corners(img)

    l = round(sqrt(l2(*tl, *tr)))

    slope = (tr[1] - tl[1]) / (tr[0] - tl[0])
    angle = np.degrees(np.arctan(slope))
    
    rotation_matrix = cv.getRotationMatrix2D((int(tl[0]), int(tl[1])), angle, 1)
    img = cv.warpAffine(img, rotation_matrix, (int(img.shape[1]), int(img.shape[0])))

    img = img[tl[1]: tl[1] + l, tl[0]: tl[0] + l, :]

    return img

# task_1 specific

def solve_task_1(img_i, identify_digits = False):

    img = get_center(img_i)

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

            d = cv.imread(f"{TEMPLATE_PATH}digit_{i}_clasic.jpg")
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
        save_solution(img_i, _with_digits(), digits = True, jigsaw = False)
    else:
        save_solution(img_i, _without_digits(), digits = False, jigsaw = False)

# task_2 specific

def get_borders(img):

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    l = img.shape[0]
    chunk_l = l // 9

    def _check_line(x1, y1, x2, y2):

        d0 = abs(x1 - x2)
        d1 = abs(y1 - y2)

        if (d0 > chunk_l // 5) and (d1 > chunk_l // 5):
            return False

        return True
       
    image_m_blur = cv.medianBlur(img, 7)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 11) 
    image_sharpened = cv.addWeighted(image_m_blur, 1.5, image_g_blur, -1, 0)
    _, thresh = cv.threshold(image_sharpened, 10, 255, cv.THRESH_BINARY)

    edges =  cv.Canny(thresh, 150, 400)
    lines = cv.HoughLinesP(image=edges, rho=1, theta=np.pi/180, threshold=25, minLineLength = 0, maxLineGap = chunk_l // 3)
    
    edges = np.zeros(edges.shape)
    
    for line in lines:

        x1, y1, x2, y2 = line[0]

        if _check_line(x1, y1, x2, y2):
            cv.line(edges, (x1, y1), (x2, y2), (255, 255, 255), 3)

    edges = edges.astype(np.uint8)

    horizontal_border = cv.imread(f"{TEMPLATE_PATH}low_edge.jpg")
    horizontal_border = cv.cvtColor(horizontal_border, cv.COLOR_BGR2GRAY)
    horizontal_border = cv.resize(horizontal_border, (int(0.7 * chunk_l), int(0.6 * chunk_l)))

    vertical_border = cv.imread(f"{TEMPLATE_PATH}up_edge.jpg")
    vertical_border = cv.cvtColor(vertical_border, cv.COLOR_BGR2GRAY)
    vertical_border = cv.resize(vertical_border, (int(0.6 * chunk_l), int(0.7 * chunk_l)))
    
    def _check_horizontal_border(chunk):
        
        match = cv.matchTemplate(chunk, horizontal_border, cv.TM_SQDIFF_NORMED)
        minval, _, _, _ = cv.minMaxLoc(match)

        if minval < 0.95:
            return True

        return False

    def _check_vertical_border(chunk):
        
        match = cv.matchTemplate(chunk, vertical_border, cv.TM_SQDIFF_NORMED)
        minval, _, _, _ = cv.minMaxLoc(match)

        if minval < 0.95:
            return True

        return False

    # chunk_border[i][j] = [border UP, border, DOWN, border LEFT, border RIGHT]
    chunk_border = [[[False, False, False, False] for _ in range(9)] for _ in range(9)]

    for k in range(9):

        chunk_border[0][k][0] = True
        chunk_border[8][k][1] = True
        chunk_border[k][0][2] = True
        chunk_border[k][8][3] = True

    # horizontal borders

    for i in range(9 - 1):
        for j in range(9):

            off_i = i
            off_j = j

            chunk = edges[i * chunk_l + off_i + chunk_l // 2: (i + 1) * chunk_l + off_i + chunk_l // 2,
                            j * chunk_l + off_j: (j + 1) * chunk_l + off_j]

            is_border = _check_horizontal_border(chunk)
            if is_border:

                chunk_border[i][j][1] = True
                chunk_border[i + 1][j][0] = True

    # vertical borders

    for i in range(9):
        for j in range(9 - 1):

            off_i = i
            off_j = j

            chunk = edges[i * chunk_l + off_i: (i + 1) * chunk_l + off_i,
                            j * chunk_l + off_j + chunk_l // 2: (j + 1) * chunk_l + off_j + chunk_l // 2]

            is_border = _check_vertical_border(chunk)
            if is_border:
                
                chunk_border[i][j][3] = True
                chunk_border[i][j + 1][2] = True

    return chunk_border

def fill_regions(chunk_borders):

    # region_matrix[i][j] - region_number
    region_matrix = [[0 for _ in range(9)] for _ in range(9)]

    #region_freq = [0 for i in range(10)]

    u = [0, -1, 0, 1]
    v = [1, 0, -1, 0]
    b = [3, 0, 2, 1]

    def _fill(i, j, reg):

        nonlocal region_matrix

        region_matrix[i][j] = reg
        #region_freq[reg] += 1

        for mov in range(4):

            i_ = i + u[mov]
            j_ = j + v[mov]

            if (0 <= i_ < 9) and (0 <= j_ < 9) and (region_matrix[i_][j_] == 0) and (chunk_borders[i][j][b[mov]] is False):
                _fill(i_, j_, reg)

    next_region = 1

    for i in range(9):
        for j in range(9):
            
            if region_matrix[i][j] == 0:

                _fill(i, j, next_region)
                next_region += 1

    '''for i in range(1, 10):
        print(region_freq[i])
        assert(region_freq[i] == 9)'''

    return region_matrix

def get_symbols(img, identify_digits = False):

    l = img.shape[0]
    chunk_l = l // 9

    empty = np.full((int(chunk_l * 0.7), int(chunk_l * 0.7)), np.uint8(255))

    def _j_type(p):

        j_type = "color"
        ps = [int(p[0]), int(p[1]), int(p[2])]
        ps.sort()

        if ps[1] - ps[0] < 7 and ps[2] - ps[0] < 7 and ps[2] - ps[1]  < 7:
            j_type = "gray"

        return j_type

    j_type = _j_type(img[chunk_l // 4, chunk_l // 4])
        
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

            d = cv.imread(f"{TEMPLATE_PATH}digit_{i}_j{j_type}.jpg")

            if j_type == "color":
                d = cv.resize(d, (chunk_l // 2, chunk_l // 2))
            else:
                d = cv.resize(d, (chunk_l * 3 // 5, chunk_l * 3 // 5))

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

def solve_task_2(img_i, identify_digits = False):

    img = get_center(img_i, jigsaw = True)

    region_matrix = fill_regions(get_borders(img))
    symbol_matrix = get_symbols(img, identify_digits)

    res_matrix = [[None for _ in range(9)] for _ in range(9)]
    for i in range(9):
        for j in range(9):
            
            res_matrix[i][j] = f"{region_matrix[i][j]}{symbol_matrix[i][j]}"

    if identify_digits:
        save_solution(img_i, res_matrix, digits = True, jigsaw = True)
    else:
        save_solution(img_i, res_matrix, digits = False, jigsaw = True)

# --------------

if __name__ == "__main__":

    print(f"[i] inceperea rezolvarii taskului 1...")
    
    for img_i in range(1, CLASIC_IMG_CNT + 1):
        
        try:
            solve_task_1(img_i, identify_digits = False)
            print(f"[*] ---- imagine {img_i}, task 1, finalizat")

        except Exception as err:
            print(f"[!] eroare {err} la procesarea imaginii clasice {img_i}, outputul fara predictiile cifrelor nu a fost generat")

        try:
            solve_task_1(img_i, identify_digits = True)
            print(f"[*] ---- imagine {img_i}, task 1 bonus, finalizat")

        except Exception as err:
            print(f"[!] eroare {err} la procesarea imaginii clasice {img_i}, outputul cu predictiile cifrelor nu a fost generat")

    print(f"[*] task 1 finalizat")
    print(f"[i] inceperea rezolvarii taskului 2...")

    for img_i in range(1, JIGSAW_IMG_CNT + 1):
        
        try:
            solve_task_2(img_i, identify_digits = False)
            print(f"[*] ---- imagine {img_i}, task 2, finalizat")

        except Exception as err:
            print(f"[!] eroare {err} la procesarea imaginii jigsaw {img_i}, outputul fara cifrelor nu a fost generat")

        try:
            solve_task_2(img_i, identify_digits = True)
            print(f"[*] ---- imagine {img_i}, task 2 bonus, finalizat")

        except Exception as err:
            print(f"[!] eroare {err} la procesarea imaginii jigsaw {img_i}, outputul cu predictiile cifrelor nu a fost generat")

    print(f"[*] task 2 finalizat")

    '''check_task_1()
    check_task_2()'''