import os

import cv2
import numpy as np


def nothing(x):
    pass


image_x, image_y = 64, 64


def create_folder(folder_name):
    if not os.path.exists('./mydata/training_set/' + folder_name):
        os.mkdir('./mydata/training_set/' + folder_name)
    if not os.path.exists('./mydata/test_set/' + folder_name):
        os.mkdir('./mydata/test_set/' + folder_name)


def capture_images(ges_name, img_path, save_folder, training_set_image_name, test_set_image_name):
    create_folder(str(ges_name))

    # cam = cv2.VideoCapture(0)

    # cv2.namedWindow("test")

    # img_counter = 0
    # t_counter = 1
    # listImage = [1, 2, 3, 4, 5]

    cv2.namedWindow("Trackbars")

    # cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    # cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    # cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    # cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    # cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    # cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

    l_h = 0
    l_s = 0
    l_v = 150
    u_h = 179
    u_s = 255
    u_v = 255

    # for loop in listImage:
    #     while True:

    # ret, frame = cam.read()
    # frame = cv2.flip(frame, 1)

    # l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    # l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    # l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    # u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    # u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    # u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    #
    # img = cv2.rectangle(frame, (800, 50), (1250, 450), (0, 255, 0), thickness=5, lineType=8, shift=0)

    img = cv2.imread(img_path)
    # img = Image.open(img_path)
    # img.show()
    # img = cv2.rectangle(img, (5, 5), (220, 220), (0, 255, 0), thickness=5, lineType=8, shift=0)
    # cv2.imshow('rhis', img)
    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    # imcrop = img[50:450, 800:1250]
    # imcrop = img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv", hsv)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    cv2.imshow("mask", mask)
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("result", result)

    # cv2.putText(frame, str(img_counter), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
    # cv2.imshow("test", frame)

    # if cv2.waitKey(1) == ord('c'):

    # if t_counter <= 350:
    if save_folder == 'train':
        img_name = "./mydata/training_set/" + str(ges_name) + "/{}.png".format(training_set_image_name)
        save_img = cv2.resize(mask, (image_x, image_y))
        cv2.imwrite(img_name, save_img)
        print("{} written!".format(img_name))
        # training_set_image_name += 1

        # if t_counter > 350 and t_counter <= 400:
    elif save_folder == 'test':
        img_name = "./mydata/test_set/" + str(ges_name) + "/{}.png".format(test_set_image_name)
        save_img = cv2.resize(mask, (image_x, image_y))
        cv2.imwrite(img_name, save_img)
        print("{} written!".format(img_name))

    cv2.waitKey(100)
    cv2.destroyAllWindows()
    # test_set_image_name += 1

    # if test_set_image_name > 250:
    # break

    # t_counter += 1
    # if t_counter == 401:
    #     t_counter = 1
    # img_counter += 1

    # elif cv2.waitKey(1) == 27:
    #     break
#     print(t_counter)
# # if test_set_image_name > 250:
# #     break

# cam.release()
# cv2.destroyAllWindows()


# capture_images(ges_name, img_path)
