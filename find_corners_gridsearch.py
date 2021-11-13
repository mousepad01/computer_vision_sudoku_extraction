import cv2 as cv
import numpy as np

def show_image(title, image):

    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def get_img(img_i):

    imgname = img_i
    if img_i < 10:
        imgname = f"0{img_i}"

    img = cv.imread(f"antrenare/clasic/{imgname}.jpg")
    img = cv.resize(img, (0, 0), fx = 0.2, fy = 0.2)

    return img

def find_corners_p(image, show, m_ker_size, g_sigma, sharp_alpha, sharp_beta, to_binary_thrsh, erode_ker_size, canny_thr1, canny_thr2):

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    image_m_blur = cv.medianBlur(image, m_ker_size)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), g_sigma) 
    image_sharpened = cv.addWeighted(image_m_blur, sharp_alpha, image_g_blur, sharp_beta, 0)
    _, thresh = cv.threshold(image_sharpened, to_binary_thrsh, 255, cv.THRESH_BINARY)

    kernel = np.ones((erode_ker_size, erode_ker_size), np.uint8)
    thresh = cv.erode(thresh, kernel)

    if show is True:

        #show_image("median blurred", image_m_blur)
        #show_image("gaussian blurred", image_g_blur)
        #show_image("sharpened", image_sharpened)    
        #show_image("threshold of blur", thresh)
        pass
    
    edges =  cv.Canny(thresh, canny_thr1, canny_thr2)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0

    top_left, top_right, bottom_left, bottom_right = None, None, None, None
   
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
    
    if show is True:

        image_copy = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)

        cv.circle(image_copy,tuple(top_left), 4, (0, 0, 255), -1)
        cv.circle(image_copy,tuple(top_right), 4, (0, 0, 255), -1)
        cv.circle(image_copy,tuple(bottom_left), 4, (0, 0, 255), -1)
        cv.circle(image_copy,tuple(bottom_right), 4, (0, 0, 255), -1)

        show_image("detected corners",image_copy)
    
    return top_left, top_right, bottom_left, bottom_right

def gridsearch():

    f = open("gridsearch.txt", "w+")

    def _l2(x0, y0, x1, y1):
        return (y1 - y0) ** 2 + (x1 - x0) ** 2

    hyperparam_val =    {
                        "m_ker_size": [3, 5], 
                        "g_sigma": [3, 5], 
                        "sharp_alpha": [1, 1.25, 1.45, 1.6], 
                        "sharp_beta": [-1.2, -0.85, -0.7, -0.5, -0.25], 
                        "to_binary_thrsh": [10, 25, 40], 
                        "erode_ker_size": [3, 5], 
                        "canny_thr1": [150],
                        "canny_thr2": [400]
                        }
    hyperparam_names = [name for name in hyperparam_val.keys()]

    def _get_hyperparam_seq(params):

        if len(params) == 0:
            yield {}
        
        else:

            for val in hyperparam_val[params[0]]:
                for seq in _get_hyperparam_seq(params[1:]):

                    to_yield = {params[0]: val} 
                    to_yield.update(seq.copy())

                    yield to_yield

    ERR_SQUARE = 0.1

    imgs = []

    IMG_CNT = 20
    for i in range(1, IMG_CNT + 1):
        imgs.append(get_img(i))

    for hyperparams in _get_hyperparam_seq(hyperparam_names):

        success_rate = 0

        IMG_CNT = 20
        for i in range(IMG_CNT):
            img = imgs[i]

            tl, tr, bl, br = find_corners_p(img, show = False, **hyperparams)
            if (tl is not None) and (tr is not None) and (bl is not None) and (br is not None):
            
                dist = [_l2(*tl, *tr), _l2(*tl, *bl), _l2(*tr, *br), _l2(*bl, *br)]

                passed = True
                for j in range(4):
                    passed = passed and (dist[j] > (img.shape[0] // 3) ** 2) and (dist[j] < (1 + ERR_SQUARE) * dist[0]) and (dist[j] > (1 - ERR_SQUARE) * dist[0])

                if passed:
                    success_rate += 1

        print(f"{hyperparams} - {success_rate}")

        if success_rate == IMG_CNT:
            print(f"{hyperparams}", file = f)

    f.flush()
    f.close()

def check_img(img_i, params):

    def _l2(x0, y0, x1, y1):
        return (y1 - y0) ** 2 + (x1 - x0) ** 2

    ERR_SQUARE = 0.1

    img = get_img(img_i)
    tl, tr, bl, br = find_corners_p(img, show = True, **params)

    print(tl, tr, bl, br)
    if (tl is not None) and (tr is not None) and (bl is not None) and (br is not None):

        dist = [_l2(*tl, *tr), _l2(*tl, *bl), _l2(*tr, *br), _l2(*bl, *br)]
        print(dist)

        success = True
        for j in range(4):
            success &= (dist[j] > (img.shape[0] // 3) ** 2) and (dist[j] < (1 + ERR_SQUARE) * dist[0]) and (dist[j] > (1 - ERR_SQUARE) * dist[0])
        
        print(f"success {success}")

if __name__ == "__main__":

    gridsearch()

    #for i in range(1, 21):
    #    check_img(i, {'m_ker_size': 3, 'g_sigma': 5, 'sharp_alpha': 1.25, 'sharp_beta': -0.7, 'to_binary_thrsh': 25, 'erode_ker_size': 3, 'canny_thr1': 150, 'canny_thr2': 400})