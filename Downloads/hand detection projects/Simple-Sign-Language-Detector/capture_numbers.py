import os

from capture import capture_images


def function():
    data_dir_list = ['test', 'train']
    data_dir_path = '/Users/tanmeshmishra/Downloads/fingers'
    sign_names_dict = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}
    sign_names_dict_cnt_test = {'zero': 0, 'one': 0, 'two': 0, 'three': 0, 'four': 0, 'five': 0}
    sign_names_dict_cnt_train = {'zero': 0, 'one': 0, 'two': 0, 'three': 0, 'four': 0, 'five': 0}
    training_set_image_name = 0
    test_set_image_name = 0
    for dataset in data_dir_list:
        img_file_list = os.listdir(data_dir_path + '/' + dataset + '/')
        for img_file in img_file_list:
            img_gesture = sign_names_dict[int(img_file[-6])]
            img_path = data_dir_path + '/' + dataset + '/' + img_file
            if dataset == 'test':
                sign_names_dict_cnt_test[img_gesture] += 1
                test_set_image_name = sign_names_dict_cnt_test[img_gesture]
            else:
                sign_names_dict_cnt_train[img_gesture] += 1
                training_set_image_name = sign_names_dict_cnt_train[img_gesture]
            capture_images(img_gesture, img_path, dataset, training_set_image_name, test_set_image_name)
function()
