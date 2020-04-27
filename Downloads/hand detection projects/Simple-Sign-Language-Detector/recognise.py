import cv2
import numpy as np


def nothing(x):
    pass


image_x, image_y = 64, 64

from keras.models import load_model

# classifier = load_model('Trained_model.h5')
classifier = load_model('Trained_model_alpanumeric.h5')


def get_img(result):
    if result[0][0] == 1:
        return 'A'
    elif result[0][1] == 1:
        return 'B'
    elif result[0][2] == 1:
        return 'C'
    elif result[0][3] == 1:
        return 'D'
    elif result[0][4] == 1:
        return 'E'
    elif result[0][5] == 1:
        return 'F'
    elif result[0][6] == 1:
        return 'G'
    elif result[0][7] == 1:
        return 'H'
    elif result[0][8] == 1:
        return 'I'
    elif result[0][9] == 1:
        return 'J'
    elif result[0][10] == 1:
        return 'K'
    elif result[0][11] == 1:
        return 'L'
    elif result[0][12] == 1:
        return 'M'
    elif result[0][13] == 1:
        return 'N'
    elif result[0][14] == 1:
        return 'O'
    elif result[0][15] == 1:
        return 'P'
    elif result[0][16] == 1:
        return 'Q'
    elif result[0][17] == 1:
        return 'R'
    elif result[0][18] == 1:
        return 'S'
    elif result[0][19] == 1:
        return 'T'
    elif result[0][20] == 1:
        return 'U'
    elif result[0][21] == 1:
        return 'V'
    elif result[0][22] == 1:
        return 'W'
    elif result[0][23] == 1:
        return 'X'
    elif result[0][24] == 1:
        return 'Y'
    elif result[0][25] == 1:
        return 'Z'
    elif result[0][26] == 1:
        return '0'
    elif result[0][27] == 1:
        return '1'
    elif result[0][28] == 1:
        return '2'
    elif result[0][29] == 1:
        return '3'
    elif result[0][30] == 1:
        return '4'
    elif result[0][31] == 1:
        return '5'


def predictor():
    import numpy as np
    from keras.preprocessing import image
    test_image = image.load_img('1.png', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)

    print(result)
    print(type(result))
    img_text = get_img(result)

    return img_text


cam = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 75, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

img_counter = 0

winname_test = "Test"
cv2.namedWindow(winname_test)
cv2.moveWindow(winname_test, 40, 800)

winname_mask = "Mask"
cv2.namedWindow(winname_mask)
cv2.moveWindow(winname_mask, 800, 30)

img_text = ''
sentence_text = ''
while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    img = cv2.rectangle(frame, (800, 50), (1250, 450), (0, 255, 0), thickness=3, lineType=8, shift=0)

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    imcrop = img[50:450, 800:1250]
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.rectangle(frame, (30, 550), (410, 610), (0, 0, 0), -1)
    cv2.putText(frame, 'Prediction : ' + str(img_text), (30, 600), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255))
    cv2.rectangle(frame, (30, 650), (330 + (40 * len(sentence_text)), 710), (0, 0, 0), -1)
    cv2.putText(frame, 'Sentence : ' + str(sentence_text), (30, 700), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))

    for contour in contours:
        cv2.drawContours(imcrop, contour, -1, (0, 255, 0), 3)
    cv2.imshow(winname_test, frame)
    cv2.imshow(winname_mask, mask)

    img_name = "1.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    img_text = predictor()

    k = cv2.waitKey(1)

    if k % 256 == 32:
        # SPACE pressed
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_text))
        sentence_text += img_text
        img_counter += 1
    elif k % 256 == ord('a'):
        # a pressed
        sentence_text += ' '
    elif k % 256 == ord('d'):
        # d pressed
        sentence_text = sentence_text[:-1]
    elif k == 27:
        # Esc key to stop
        break

cam.release()
cv2.destroyAllWindows()
