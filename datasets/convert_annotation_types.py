# Thomas Chia i-Sight Convert YOLO Annotations to PascalVOC XML
# Run Script:
'''
    cd datasets
    pip install PILLOW
    python convert_annotation_types.py --datadir PATH-TO-DATASET-DIR --labels PATH-TO-LABELS.txt
'''
# Notes:
#   The dataset folder should contain the labels and images
#   Description of YOLO Format values
#       15 0.448743 0.529142 0.051587 0.021081
#       class_number x y width height


import os
import re
import argparse

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir")
    parser.add_argument("--labels")
    args = parser.parse_args()

    folder_holding_yolo_files = args.datadir
    yolo_class_list_file = args.labels

    return folder_holding_yolo_files, yolo_class_list_file


def read_class_file(yolo_class_list_file):
    with open(yolo_class_list_file) as f:
        yolo_classes = f.readlines()
    array_of_yolo_classes = [x.strip() for x in yolo_classes]
    return array_of_yolo_classes

def is_number(n):
    try:
        float(n)
        return True
    except ValueError:
        return False


def convert_to_xml(folder_holding_yolo_files, array_of_yolo_classes):

    if not os.path.exists(folder_holding_yolo_files + os.sep + 'XML'):
    # If an XML folder does not already exist, make one
        os.mkdir('XML')

    for each_yolo_file in os.listdir(folder_holding_yolo_files):
        full_each_yolo_file = os.path.join(folder_holding_yolo_files, each_yolo_file)  
        if full_each_yolo_file.endswith("txt"):
            the_file = open(full_each_yolo_file, 'r')
            all_lines = the_file.readlines()
            image_name = each_yolo_file

            # Check to see if there is an image that matches the txt file
            if os.path.exists(full_each_yolo_file.replace('txt', 'jpeg')):
                image_name = each_yolo_file.replace('txt', 'jpeg')
            if os.path.exists(full_each_yolo_file.replace('txt', 'jpg')):
                image_name = each_yolo_file.replace('txt', 'jpg')
            if os.path.exists(full_each_yolo_file.replace('txt', 'png')):
                image_name = each_yolo_file.replace('txt', 'png')

            if not image_name == each_yolo_file:
                # If the image name is the same as the yolo filename
                # then we did NOT find an image that matches, and we will skip this code block
                orig_img = Image.open(os.path.join(folder_holding_yolo_files, image_name)) # open the image
                image_width = orig_img.width
                image_height = orig_img.height

                # Start the XML file
                with open('XML' + os.sep + each_yolo_file.replace('txt', 'xml'), 'w') as f:
                    f.write('<annotation>\n')
                    f.write('\t<folder>XML</folder>\n')
                    f.write('\t<filename>' + image_name + '</filename>\n')
                    f.write('\t<path>' + os.getcwd() + os.sep + image_name + '</path>\n')
                    f.write('\t<source>\n')
                    f.write('\t\t<database>Unknown</database>\n')
                    f.write('\t</source>\n')
                    f.write('\t<size>\n')
                    f.write('\t\t<width>' + str(image_width) + '</width>\n')
                    f.write('\t\t<height>' + str(image_height) + '</height>\n')
                    f.write('\t\t<depth>3</depth>\n') # assuming a 3 channel color image (RGB)
                    f.write('\t</size>\n')
                    f.write('\t<segmented>0</segmented>\n')
                
                    for each_line in all_lines:
                        # regex to find the numbers in each line of the text file
                        yolo_array = re.split("\s", each_line.rstrip()) # remove any extra space from the end of the line

                        # initalize the variables
                        class_number = 0
                        x_yolo = 0.0
                        y_yolo = 0.0
                        yolo_width = 0.0
                        yolo_height = 0.0
                        yolo_array_contains_only_digits = True

                        # make sure the array has the correct number of items
                        if len(yolo_array) == 5:
                            for each_value in yolo_array:
                                # If a value is not a number, then the format is not correct, return false
                                if not is_number(each_value):
                                    yolo_array_contains_only_digits = False
                            
                            if yolo_array_contains_only_digits:
                                # assign the variables
                                class_number = int(yolo_array[0])
                                object_name = array_of_yolo_classes[class_number]
                                x_yolo = float(yolo_array[1])
                                y_yolo = float(yolo_array[2])
                                yolo_width = float(yolo_array[3])
                                yolo_height = float(yolo_array[4])

                                # Convert Yolo Format to Pascal VOC format
                                box_width = yolo_width * image_width
                                box_height = yolo_height * image_height
                                x_min = str(int(x_yolo * image_width - (box_width / 2)))
                                y_min = str(int(y_yolo * image_height - (box_height / 2)))
                                x_max = str(int(x_yolo * image_width + (box_width / 2)))
                                y_max = str(int(y_yolo * image_height + (box_height / 2)))

                                # write each object to the file
                                f.write('\t<object>\n')
                                f.write('\t\t<name>' + object_name + '</name>\n')
                                f.write('\t\t<pose>Unspecified</pose>\n')
                                f.write('\t\t<truncated>0</truncated>\n')
                                f.write('\t\t<difficult>0</difficult>\n')
                                f.write('\t\t<bndbox>\n')
                                f.write('\t\t\t<xmin>' + x_min + '</xmin>\n')
                                f.write('\t\t\t<ymin>' + y_min + '</ymin>\n')
                                f.write('\t\t\t<xmax>' + x_max + '</xmax>\n')
                                f.write('\t\t\t<ymax>' + y_max + '</ymax>\n')
                                f.write('\t\t</bndbox>\n')
                                f.write('\t</object>\n')

                    # Close the annotation tag once all the objects have been written to the file
                    f.write('</annotation>\n')
                    f.close() # Close the file

if __name__ == "__main__":
    folder_holding_yolo_files, yolo_class_list_file = parse_args()
    array_of_yolo_classes = read_class_file(yolo_class_list_file)
    convert_to_xml(folder_holding_yolo_files, array_of_yolo_classes)
    # Check to make sure the sprite file is now in the folder
    if os.path.exists("XML"):
        print("Conversion complete")
    else:
        print("There was a problem converting the files")