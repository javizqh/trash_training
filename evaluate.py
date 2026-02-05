#!/usr/bin/env python3
import os
import re
from PIL import Image, ExifTags
import yaml
import shutil
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np


def tryint(s):
    try:
        return int(s)
    except Exception:
        return s


def alphanum_key(s):
    """Turn a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split("([0-9]+)", s)]


def get_images(dir, fraction=1):
    files = [x for x in os.listdir(dir) if os.path.isfile(os.path.join(dir, x))]

    files.sort(key=alphanum_key)

    fr = int(1 / fraction)
    files = [files[x] for x in range(len(files)) if x % fr == 0]
    return files


def load_label(file):
    with open(file) as f:
        data = f.readlines()
    return data


def img_rotate(im: Image) -> Image:
    # Obtain Exif orientation tag code
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == "Orientation":
            break

    if im._getexif():
        exif = dict(im._getexif().items())
        # Rotate portrait and upside down images if necessary
        if orientation in exif:
            if exif[orientation] == 3:
                return im.rotate(180, expand=True)
            if exif[orientation] == 6:
                return im.rotate(270, expand=True)
            if exif[orientation] == 8:
                return im.rotate(90, expand=True)
    return im


def load_img(file):
    im = Image.open(file)
    im = img_rotate(im)
    return im


orig_classes = {
    0: "Aluminium foil",
    1: "Battery",
    2: "Blister pack",
    3: "Bottle",
    4: "Bottle cap",
    5: "Broken glass",
    6: "Can",
    7: "Carton",
    8: "Cup",
    9: "Food waste",
    10: "Glass jar",
    11: "Lid",
    12: "Other plastic",
    13: "Paper",
    14: "Paper bag",
    15: "Plastic bag & wrapper",
    16: "Plastic container",
    17: "Plastic glooves",
    18: "Plastic utensils",
    19: "Pop tab",
    20: "Rope & strings",
    21: "Scrap metal",
    22: "Shoe",
    23: "Squeezable tube",
    24: "Straw",
    25: "Styrofoam piece",
    26: "Unlabeled litter",
    27: "Cigarette",
}

if __name__ == "__main__":
    src = "./datasets/TACO"
    model = YOLO("New-Results/TACO2/weights/best.pt")
    yaml_path = os.path.join(src, f"tmp.yaml")

    data = {
        "path": src,
        "train": "images/tmp/",
        "val": "images/tmp/",
        "names": orig_classes,
    }

    if os.path.exists(os.path.join(src, "images", "tmp")):
        shutil.rmtree(os.path.join(src, "images", "tmp"))

    if os.path.exists(os.path.join(src, "labels", "tmp")):
        shutil.rmtree(os.path.join(src, "labels", "tmp"))

    os.mkdir(os.path.join(src, "images", "tmp"))
    os.mkdir(os.path.join(src, "labels", "tmp"))

    # Writing the data to a YAML file
    with open(yaml_path, "w") as file:
        yaml.dump(data, file)

    for split in ["val"]:
        labels_column = []
        total_labels = []
        image_names_column = []
        area_list = [[None]] * len(orig_classes)

        images_src = os.path.join(src, "images", split)
        labels_src = os.path.join(src, "labels", split)
        images_out = os.path.join(src, "images", "tmp")
        labels_out = os.path.join(src, "labels", "tmp")

        files = get_images(images_src)

        for f in files:
            if f.endswith(".jpg"):
                filename, extension = os.path.splitext(f)

                img = os.path.join(images_src, filename + extension)
                label = os.path.join(labels_src, filename + ".txt")
                new_img = os.path.join(images_out, "tmp" + extension)
                new_label = os.path.join(labels_out, "tmp" + ".txt")

                im = load_img(img)
                data = load_label(label)

                im.save(new_img)
                shutil.copyfile(label, new_label)

                width, height = im.size
                try:
                    metrics = model.val(data=yaml_path, plots=False)
                except Exception:
                    continue
                for line in data:
                    [name, x, y, w, h] = line.rstrip().split(" ")

                    pixel_area = int(float(w) * width * float(h) * height)
                    mAp = metrics.box.maps[int(name)]
                    if area_list[int(name)] == [None]:
                        area_list[int(name)] = [(pixel_area, mAp)]
                    else:
                        area_list[int(name)].append((pixel_area, mAp))
        index = -1
        for cl in area_list:
            index += 1
            if cl == [None]:
                continue
            cl.sort(key=lambda x: x[0], reverse=True)
            area = []
            for i in range(len(cl)):
                area.append(cl[i][0])
            hist, bins = np.histogram(area, bins=20)
            base = 0
            boxplots = []
            for h in hist:
                mAp = []
                for i in range(h):
                    mAp.append(cl[base + i][1])
                base += h
                boxplots.append(mAp)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(
                boxplots,
            )
            ax.plot(np.linspace(1, 20, 20), hist / np.sum(hist), "r+")
            ax.set_xticklabels(bins[:-1], rotation=90)
            ax.set_title(f"{orig_classes[index]}")
            ax.set_xlabel("Area of instance: in pixels")
            ax.set_ylabel("mAp 0.5-0.95")
            plt.savefig(f"area/{orig_classes[index]}", bbox_inches="tight")
            plt.close()
