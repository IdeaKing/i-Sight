# Thomas Chia. 2022
# Converts PascalVOC to RetinaNet .CSV
# https://github.com/fizyr/keras-retinanet#csv-datasets

import os
import csv
import argparse
import xml.etree.ElementTree as ET


def parse_process_voc(path_to_xml):
    """Parses the PascalVOC XML Type file."""
    source = open(path_to_xml)
    root = ET.parse(source).getroot()
    filename = str(root.findtext("filename"))
    boxes = root.findall("object")
    image_size = (int(root.findtext("size/width")),
                    int(root.findtext("size/height")))

    bboxes = []
    labels = []

    for b in boxes:
        bb = b.find("bndbox")
        bb = [(float(bb.findtext("xmin"))/image_size[0]),
              (float(bb.findtext("ymin"))/image_size[1]),
              (float(bb.findtext("xmax"))/image_size[0]),
              (float(bb.findtext("ymax"))/image_size[1])]
        bboxes.append(bb)
        labels.append(str(b.findtext("name")))
    source.close()
    return labels, bboxes, filename


def write_to_csv(writer, labels, bboxes, filename):
    for label, bboxs in zip(labels, bboxes):
        xmin, ymin, xmax, ymax = bboxs
        new_line = [filename, xmin, ymin, xmax, ymax, label]
        writer.writerow(new_line)


def main(path_to_label_dir, csv_save_path):
    print(f"Converting {path_to_label_dir} to csv file.")
    with open(csv_save_path, "x", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        for label_file in os.listdir(path_to_label_dir):
            label_path = os.path.join(path_to_label_dir, label_file)
            # print(f"Reading {label_path}")
            labels, bboxes, filename = parse_process_voc(label_path)
            write_to_csv(csv_writer, labels, bboxes, filename)
    csvfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PascalVOC to RetinaNet .CSV")
    parser.add_argument("--label-dir",
                        help="Path to the label directory",
                        required=True)
    parser.add_argument("--save-csv-path",
                        help="Path to save the csv.",
                        required=True)
    args = parser.parse_args()

    main(args.label_dir, args.save_csv_path)
