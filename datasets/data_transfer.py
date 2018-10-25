import os
import xml.etree.ElementTree as et
import cv2

object_name_dict = {
        'rice': '1',
        'soup': '2',
        'rect': '3',
        'lcir': '4',
        'ssquare': '5',
        'msquare': '6',
        'lsquare': '7',
        'bsquare': '8',
        'ellipse': '9'
    }

def data_transfer(xml_path, img_dir, save_dir, index):
    tree = et.parse(xml_path)
    root = tree.getroot()
    file_name = root.find('filename')
    file_name_text = os.path.join(img_dir, file_name.text)
    my_object = root.findall('object')
    count = 0
    for single_object in my_object:
        single_object_name = single_object.find('name').text
        label = object_name_dict[single_object_name]
        single_object_rect = single_object.find('bndbox')
        x_min = int(single_object_rect.find('xmin').text)
        x_max = int(single_object_rect.find('xmax').text)
        y_min = int(single_object_rect.find('ymin').text)
        y_max = int(single_object_rect.find('ymax').text)
        image = cv2.imread(file_name_text)
        cropImg = image[y_min:y_max, x_min:x_max]
        cv2.imwrite(os.path.join(save_dir, str(index + count)+"_"+label+".png"), cropImg)
        print("save image to "+ os.path.join(save_dir, str(index + count)+"_"+label+".png"))
        count = count + 1
    return count

if __name__ == "__main__":
    img_dir = "../datas/datas/img/"
    xml_dir = "../datas/datas/xml/"
    xml_list = os.listdir(xml_dir)
    list_label = []
    index = 10000
    for xml_path in xml_list:
        file_path = os.path.join(xml_dir, xml_path)
        save_dir = "../datas/transdatas/train/"
        num = data_transfer(file_path, img_dir, save_dir, index)
        index = index + num