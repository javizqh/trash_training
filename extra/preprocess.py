import collections
import os
from PIL import Image, ExifTags
import re
import sys
import matplotlib

# matplotlib.use("TKAgg")
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd


def clamp(n, min, max):
    if n < min:
        return min
    elif n > max:
        return max
    else:
        return n


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


class Preprocess:
    def __init__(self, img_h=640, img_w=640, map_labels=None, n_classes=0, class_map={}):
        self.img_h = img_h
        self.img_w = img_w
        self.map_labels = map_labels
        self.n_classes = n_classes
        self.class_map = class_map


    def __load_label(self, file):
        with open(file) as f:
            data = f.readlines()
        return data

    def __load_img(self, file):
        im = Image.open(file)
        im = img_rotate(im)
        return im

    def __get_mapped_class(self, src_class):
        name = src_class
        if self.map_labels is not None:
            name = str(self.map_labels[int(name)])
        return name

    def __get_images(self, dir, fraction=1):
        files = [x for x in os.listdir(dir) if os.path.isfile(os.path.join(dir, x))]

        files.sort(key=alphanum_key)

        fr = int(1 / fraction)
        files = [files[x] for x in range(len(files)) if x % fr == 0]
        return files

    def show_distribution(self, name, title, class_list, values):
        class_distribution = []
        class_labels = []

        color = plt.cm.viridis(np.linspace(0, 1, len(class_list)))
        for i in sorted(class_list):
            try:
                # class_distribution.append(100 * np.count_nonzero(df[i]) / df.shape[0])
                class_distribution.append(values[i])
                class_labels.append(i)
            except Exception:
                class_distribution.append(0)

        # Make plot
        plt.figure(figsize=(10, 5))
        plt.bar(class_labels, class_distribution, color=color, width=1, edgecolor="k")
        plt.title(f"{title}    nImages: {str(np.sum(values))}")
        plt.xticks(rotation=90)
        plt.xlabel("Classes")
        plt.ylabel("Instances")
        plt.yticks(
            np.arange(
                0,
                max(class_distribution) + int(max(class_distribution) / 10),
                int(max(class_distribution) / 10),
            )
        )
        plt.ylim(0, max(class_distribution))
        # plt.show()
        plt.savefig(name + title, bbox_inches="tight")
        plt.close()

    def mantain(
        self,
        fig_name,
        src,
        out,
        n_classes=0,
        fraction=1,
        skew=False,
        normalize=False,
        max=0,
        equal=False,
    ):
        imgs = os.path.join(src, "images")
        labels = os.path.join(src, "labels")
        images_out = os.path.join(out, "images")
        labels_out = os.path.join(out, "labels")

        area_list = [[None]] * self.n_classes

        if equal:
            cl_instances = np.zeros(self.n_classes, dtype=int)

        if max > 0:
            cl_instances = np.zeros(
                len(np.where(np.array(self.map_labels) >= 0)[0]), dtype=int
            )
            cl_limit = 150

        for f in self.__get_images(imgs, fraction):
            if f.endswith(".jpg"):
                filename, extension = os.path.splitext(f)

                img = os.path.join(imgs, filename + extension)
                label = os.path.join(labels, filename + ".txt")
                new_img = os.path.join(images_out, filename + extension)
                new_label = os.path.join(labels_out, filename + ".txt")

                im = self.__load_img(img)
                data = self.__load_label(label)

                width, height = im.size

                im.save(new_img)

                is_empty = True

                with open(new_label, "w") as f:
                    for line in data:
                        [name, x, y, w, h] = line.rstrip().split(" ")

                        pixel_area = float(w)*float(h)

                        if normalize:
                            x = float(x) / width
                            y = float(y) / height
                            w = float(w) / width
                            h = float(h) / height

                        if skew:
                            x = float(x) + float(w) / 2
                            y = float(y) + float(h) / 2

                        if int(name) < n_classes:
                            name = self.__get_mapped_class(name)
                            if name == "-1":
                                continue

                            if max > 0:
                                if cl_instances[int(name)] >= cl_limit:
                                    continue

                            if equal or max > 0:
                                cl_instances[int(name)] += 1

                            if area_list[int(name)] == [None]:
                                area_list[int(name)] = [pixel_area]
                            else:
                                area_list[int(name)].append(pixel_area)

                            f.write(f"{name} {x} {y} {w} {h}\n")
                            is_empty = False

                if is_empty:
                    os.remove(new_img)
                    os.remove(new_label)

        for i in range(len(area_list)):
            class_area = area_list[i]
            if class_area != [None]:
                data = np.array(sorted(class_area))

                plt.figure(figsize=(10, 5))
                plt.hist(data, bins=30,range=[0,640*640], color="blue", edgecolor='black')

                # Adding labels and title
                plt.xlabel('Pixel Area')
                plt.ylabel('Instances')
                plt.title(f'Area of instances of {self.class_map[i]}')

                # Display the plot
                # plt.show()
                plt.savefig(f"{src}_Area_of_{self.class_map[i]}", bbox_inches="tight")
                plt.close()

        if equal:
            cl_limit = np.max(cl_instances)
            appendix = 0
            while np.min(cl_instances) != cl_limit:
                appendix += 1
                for f in self.__get_images(imgs, fraction):
                    if f.endswith(".jpg"):
                        filename, extension = os.path.splitext(f)

                        img = os.path.join(imgs, filename + extension)
                        label = os.path.join(labels, filename + ".txt")
                        new_img = os.path.join(
                            images_out, filename + "_" + str(appendix) + extension
                        )
                        new_label = os.path.join(
                            labels_out, filename + "_" + str(appendix) + ".txt"
                        )

                        im = self.__load_img(img)
                        data = self.__load_label(label)

                        width, height = im.size

                        im.save(new_img)

                        is_empty = True

                        with open(new_label, "w") as f:
                            for line in data:
                                [name, x, y, w, h] = line.rstrip().split(" ")

                                if normalize:
                                    x = float(x) / width
                                    y = float(y) / height
                                    w = float(w) / width
                                    h = float(h) / height

                                if skew:
                                    x = float(x) + float(w) / 2
                                    y = float(y) + float(h) / 2

                                if int(name) < n_classes:
                                    name = self.__get_mapped_class(name)
                                    if name == "-1":
                                        continue

                                    if cl_instances[int(name)] >= cl_limit:
                                        continue

                                    cl_instances[int(name)] += 1

                                    f.write(f"{name} {x} {y} {w} {h}\n")
                                    is_empty = False

                        if is_empty:
                            os.remove(new_img)
                            os.remove(new_label)

    def r_crop(
        self,
        src,
        out,
        n_classes=0,
        fraction=1,
        skew=False,
        normalize=False,
        max=0,
        equal=False,
    ):
        images_src = os.path.join(src, "images")
        labels_src = os.path.join(src, "labels")
        images_out = os.path.join(out, "images")
        labels_out = os.path.join(out, "labels")

        files = self.__get_images(images_src, fraction)

        if max > 0:
            cl_instances = np.zeros(
                len(np.where(np.array(self.map_labels) >= 0)[0]), dtype=int
            )
            cl_limit = 150
        distribution = {}

        for f in files:
            if f.endswith(".jpg"):
                filename, extension = os.path.splitext(f)

                img = os.path.join(images_src, filename + extension)
                label = os.path.join(labels_src, filename + ".txt")

                im = self.__load_img(img)
                data = self.__load_label(label)

                width, height = im.size

                max_width = int(width * self.img_w / height)
                max_height = int(height * self.img_h / width)

                new_size = (self.img_w, self.img_h)

                if width < height:
                    new_size = (self.img_h, max_height)
                elif width > height:
                    new_size = (max_width, self.img_w)

                im = im.resize(new_size)

                index = 0
                # print(cl_instances)
                for i in range(0, new_size[0], self.img_h):
                    for j in range(0, new_size[1], self.img_w):
                        new_img = os.path.join(images_out, f"{filename}_{index}.jpg")
                        new_label = os.path.join(labels_out, f"{filename}_{index}.txt")

                        start_x = (
                            i
                            if i + self.img_h < new_size[0]
                            else new_size[0] - self.img_h
                        )
                        start_y = (
                            j
                            if j + self.img_w < new_size[1]
                            else new_size[1] - self.img_w
                        )
                        im_crop = (
                            start_x,
                            start_y,
                            start_x + self.img_w,
                            start_y + self.img_h,
                        )
                        im2 = im.crop(im_crop)
                        instances = 0
                        with open(new_label, "w") as f:
                            for line in data:
                                try:
                                    [name, x, y, w, h] = line.rstrip().split(" ")
                                    if int(name) > n_classes:
                                        continue

                                    if normalize:
                                        x = float(x) / width
                                        y = float(y) / height
                                        w = float(w) / width
                                        h = float(h) / height

                                    if skew:
                                        x = float(x) + float(w) / 2
                                        y = float(y) + float(h) / 2

                                    x = int(float(x) * new_size[0] - start_x)
                                    y = int(float(y) * new_size[1] - start_y)
                                    w = int(float(w) * new_size[0])
                                    h = int(float(h) * new_size[1])

                                    if (
                                        x - w / 2 > 640
                                        or y - h / 2 > 640
                                        or x + w / 2 < 0
                                        or y + h / 2 < 0
                                    ):
                                        continue

                                    name = self.__get_mapped_class(name)

                                    if name == "-1":
                                        continue

                                    if max > 0:
                                        if cl_instances[int(name)] >= cl_limit:
                                            continue

                                        cl_instances[int(name)] += 1

                                    x = clamp(x / self.img_w, 0, 640)
                                    y = clamp(y / self.img_h, 0, 640)
                                    w = clamp(w / self.img_w, 0, 640)
                                    h = clamp(h / self.img_h, 0, 640)

                                    f.write(f"{name} {x} {y} {w} {h}\n")
                                    instances += 1
                                except Exception:
                                    pass
                        if instances == 0:
                            os.remove(new_label)
                        else:
                            im2.save(new_img)
                        index += 1

                try:
                    distribution[str(index)] += 1
                except:
                    distribution[str(index)] = 1

        a = list(distribution.keys())
        a.sort()
        sd = {i: distribution[i] for i in a}
        color = plt.cm.viridis(np.linspace(0, 1, len(sd.values())))

        # Make plot
        plt.figure(figsize=(10, 5))
        plt.bar(sd.keys(), sd.values(), color=color, width=1, edgecolor="k")
        plt.title(f"{src}    DA-Crop")
        plt.xticks(rotation=90)
        plt.xlabel("N new images")
        plt.ylabel("Instances")
        plt.yticks(np.arange(0, max(sd.values()), int(max(sd.values()) / 10)))
        plt.ylim(0, max(sd.values()))
        # plt.show()
        plt.savefig(f"{src}_DA-Crop", bbox_inches="tight")
        plt.close()

    def r_crop_splited(
        self,
        fig_name,
        src,
        out,
        n_classes=0,
        fraction=1,
        skew=False,
        normalize=False,
        max=0,
        equal=False,
    ):

        if max > 0:
            cl_instances = np.zeros(
                len(np.where(np.array(self.map_labels) >= 0)[0]), dtype=int
            )
            cl_limit = 150

        distribution = {}
        area_list = [[None]] * self.n_classes

        for split in ["train", "val", "test"]:
            labels_column = []
            total_labels = []
            image_names_column = []

            images_src = os.path.join(src, "images", split)
            labels_src = os.path.join(src, "labels", split)
            images_out = os.path.join(out, "images", split)
            labels_out = os.path.join(out, "labels", split)

            files = self.__get_images(images_src, fraction)

            for f in files:
                if f.endswith(".jpg"):
                    filename, extension = os.path.splitext(f)

                    img = os.path.join(images_src, filename + extension)
                    label = os.path.join(labels_src, filename + ".txt")

                    im = self.__load_img(img)
                    data = self.__load_label(label)

                    os.remove(img)
                    os.remove(label)

                    width, height = im.size

                    max_width = int(width * self.img_w / height)
                    max_height = int(height * self.img_h / width)

                    new_size = (self.img_w, self.img_h)

                    if width < height:
                        new_size = (self.img_h, max_height)
                    elif width > height:
                        new_size = (max_width, self.img_w)

                    im = im.resize(new_size)

                    index = 0

                    for i in range(0, new_size[0], self.img_h):
                        for j in range(0, new_size[1], self.img_w):
                            new_img = os.path.join(
                                images_out, f"{filename}_{index}.jpg"
                            )
                            new_label = os.path.join(
                                labels_out, f"{filename}_{index}.txt"
                            )

                            start_x = (
                                i
                                if i + self.img_h < new_size[0]
                                else new_size[0] - self.img_h
                            )
                            start_y = (
                                j
                                if j + self.img_w < new_size[1]
                                else new_size[1] - self.img_w
                            )
                            im_crop = (
                                start_x,
                                start_y,
                                start_x + self.img_w,
                                start_y + self.img_h,
                            )
                            im2 = im.crop(im_crop)
                            instances = 0
                            with open(new_label, "w") as f:
                                orig_labels = []
                                for line in data:
                                    try:
                                        [name, x, y, w, h] = line.rstrip().split(" ")
                                        x = int(float(x) * new_size[0] - start_x)
                                        y = int(float(y) * new_size[1] - start_y)
                                        w = int(float(w) * new_size[0])
                                        h = int(float(h) * new_size[1])

                                        if (
                                            x - w / 2 > 640
                                            or y - h / 2 > 640
                                            or x + w / 2 < 0
                                            or y + h / 2 < 0
                                        ):
                                            continue

                                        if max > 0:
                                            if cl_instances[int(name)] >= cl_limit:
                                                continue

                                        if max > 0:
                                            cl_instances[int(name)] += 1

                                        x = clamp(x / self.img_w, 0, 640)
                                        y = clamp(y / self.img_h, 0, 640)
                                        w = clamp(w / self.img_w, 0, 640)
                                        h = clamp(h / self.img_h, 0, 640)

                                        orig_labels.append(int(name))
                                        total_labels.append(int(name))

                                        if area_list[int(name)] == [None]:
                                            area_list[int(name)] = [w*h]
                                        else:
                                            area_list[int(name)].append(w*h)

                                        f.write(f"{name} {x} {y} {w} {h}\n")
                                        instances += 1
                                    except Exception as e:
                                        print(e)

                            if instances == 0:
                                os.remove(new_label)
                            else:
                                im2.save(new_img)
                                labels_column.append(orig_labels)
                                image_names_column.append(new_img)

                            index += 1

                    try:
                        distribution[str(index)] += 1
                    except:
                        distribution[str(index)] = 1

            csv_rows = zip(image_names_column, labels_column)

            # Load dataframe and convert to column per label
            df = pd.DataFrame(csv_rows, columns=["images", "labels"])
            cs = collections.Counter(total_labels)

            text_to_category = {label: [] for label in cs.keys()}
            for _, item in df.iterrows():
                for label in text_to_category:
                    text_to_category[label].append(item["labels"].count(label))

            for label in text_to_category:
                df[label] = text_to_category[label]

            del df["labels"]

            if equal:
                cl_instances = np.zeros(self.n_classes, dtype=int)
                limit = 0

                for label, val in text_to_category.items():
                    cl_instances[label] = np.count_nonzero(val)
                    if np.count_nonzero(val) > limit:
                        limit = np.count_nonzero(val)

                for series_name, series in df.items():
                    if series_name == "images":
                        continue
                    if cl_instances[series_name] == 0:
                        continue

                    try:
                        distribution[str(series_name)] = 0
                    except:
                        distribution[str(series_name)] = 0

                    existing = df[df[series_name] == 1]

                    if len(existing["images"]) == 0:
                        continue

                    n_missing = limit - cl_instances[series_name]
                    appendix = 0

                    while True:
                        sys.stdout.flush()
                        if n_missing == 0:
                            break

                        appendix += 1

                        for f in existing["images"]:

                            filename, extension = os.path.splitext(f)

                            img = filename + extension
                            label = filename.replace("/images/", "/labels/") + ".txt"
                            new_img = filename + "_" + str(appendix) + extension
                            new_label = (
                                filename.replace("/images/", "/labels/")
                                + "_"
                                + str(appendix)
                                + ".txt"
                            )

                            im = self.__load_img(img)
                            data = self.__load_label(label)

                            im.save(new_img)

                            with open(new_label, "w") as f:
                                for line in data:
                                    [name, x, y, w, h] = line.rstrip().split(" ")

                                    if int(name) != series_name:
                                        continue

                                    cl_instances[int(name)] += 1
                                    distribution[str(name)] += 1
                                    n_missing -= 1
                                    f.write(f"{name} {x} {y} {w} {h}\n")

                                    if n_missing == 0:
                                        break
                                if n_missing == 0:
                                    break
                            if n_missing == 0:
                                break

                del df["images"]

                self.show_distribution(
                    fig_name,
                    f"Equal_{split}_Split",
                    range(self.n_classes),
                    cl_instances,
                )

        a = list(distribution.keys())
        a.sort()
        sd = {i: distribution[i] for i in a}
        color = plt.cm.viridis(np.linspace(0, 1, len(sd.values())))

        # Make plot
        plt.figure(figsize=(10, 5))
        plt.bar(sd.keys(), sd.values(), color=color, width=1, edgecolor="k")
        plt.title(f"{src}    DA-Crop")
        plt.xticks(rotation=90)
        plt.xlabel("N new images")
        plt.ylabel("Instances")
        plt.yticks(
            np.arange(0, np.max(list(sd.values())), int(np.max(list(sd.values())) / 10))
        )
        plt.ylim(0, np.max(list(sd.values())))
        # plt.show()
        plt.savefig(f"{src}_DA-Crop", bbox_inches="tight")
        plt.close()

    def fill(self, src, out, n_classes=0, fraction=1, skew=False, normalize=False):
        images_src = os.path.join(src, "images")
        labels_src = os.path.join(src, "labels")
        images_out = os.path.join(out, "images")
        labels_out = os.path.join(out, "labels")

        for f in self.__get_images(images_src, fraction):
            if f.endswith(".jpg"):
                filename, extension = os.path.splitext(f)

                img = os.path.join(images_src, filename + extension)
                label = os.path.join(labels_src, filename + ".txt")
                new_img = os.path.join(images_out, filename + extension)
                new_label = os.path.join(labels_out, filename + ".txt")

                im = self.__load_img(img)
                data = self.__load_label(label)

                width, height = im.size

                min_width = int(width * self.img_w / height)
                min_height = int(height * self.img_h / width)

                new_size = (self.img_w, self.img_h)

                if width > height:
                    new_size = (self.img_w, min_height)
                elif width < height:
                    new_size = (min_width, self.img_h)

                im = im.resize(new_size)
                new_im = Image.new("RGB", (self.img_w, self.img_h), (0, 0, 0))
                new_im.paste(
                    im,
                    (
                        int((self.img_w - new_size[0]) / 2),
                        int((self.img_h - new_size[1]) / 2),
                    ),
                )
                new_im.save(new_img)

                ratio_w = new_size[0] / self.img_w
                ratio_h = new_size[1] / self.img_h

                with open(new_label, "w") as f:
                    for line in data:
                        try:
                            [name, x, y, w, h] = line.rstrip().split(" ")

                            if int(name) > n_classes:
                                continue

                            if normalize:
                                x = float(x) / width
                                y = float(y) / height
                                w = float(w) / width
                                h = float(h) / height

                            if skew:
                                x = float(x) + float(w) / 2
                                y = float(y) + float(h) / 2

                            x = int(
                                float(x) * self.img_w * ratio_w
                                + (self.img_w - new_size[0]) / 2
                            )
                            y = int(
                                float(y) * self.img_h * ratio_h
                                + (self.img_h - new_size[1]) / 2
                            )
                            w = int(float(w) * self.img_w * ratio_w)
                            h = int(float(h) * self.img_h * ratio_h)

                            name = self.__get_mapped_class(name)
                            x = x / self.img_w
                            y = y / self.img_h
                            w = w / self.img_w
                            h = h / self.img_h

                            f.write(f"{name} {x} {y} {w} {h}\n")
                        except Exception:
                            pass

    # def center_crop(self, src, out, n_classes=0, fraction=1, skew=False, normalize=False):
    #     images_src = os.path.join(src, "images")
    #     labels_src = os.path.join(src, "labels")
    #     images_out = os.path.join(out, "images")
    #     labels_out = os.path.join(out, "labels")

    #     for f in self.__get_images(images_src, fraction):
    #         if f.endswith(".jpg"):
    #             filename, extension = os.path.splitext(f)

    #             img = os.path.join(images_src, filename + extension)
    #             label = os.path.join(labels_src, filename + ".txt")

    #             im = self.__load_img(img)
    #             data = self.__load_label(label)

    #             width, height = im.size

    #             if width == height:
    #                 im = im.resize((self.img_w, self.img_h))
    #                 im.save(img)
    #                 print("No resize", img)
    #                 continue

    #             index = 0
    #             try:
    #                 for line in data:
    #                     [name, x, y, w, h] = line.rstrip().split(" ")
    #                     x = int(float(x) * width)
    #                     y = int(float(y) * height)
    #                     w = int(float(w) * width)
    #                     h = int(float(h) * height)

    #                     f_w = w
    #                     f_h = h

    #                     if w < self.img_w:
    #                         f_w = self.img_w

    #                     if h < self.img_h:
    #                         f_h = self.img_h

    #                     f_h = max(f_h, f_w)
    #                     f_w = f_h

    #                     max_increase = min(
    #                         int(x - f_w / 2),
    #                         int(y - f_h / 2),
    #                         width - int(x + f_w / 2),
    #                         height - int(y + f_h / 2),
    #                     )
    #                     f_h += max_increase * 2
    #                     f_w += max_increase * 2

    #                     final_max_bbox = (
    #                         (int(x - f_w / 2), int(y - f_h / 2)),
    #                         (int(x + f_w / 2), int(y + f_h / 2)),
    #                     )

    #                     im2 = im.crop(
    #                         (
    #                             final_max_bbox[0][0],
    #                             final_max_bbox[0][1],
    #                             final_max_bbox[1][0],
    #                             final_max_bbox[1][1],
    #                         )
    #                     )
    #                     im2 = im2.resize((640, 640))
    #                     im2.save(
    #                         os.path.join(
    #                             images_path, f"{filename}_{index}{extension}")
    #                     )
    #                     with open(
    #                         os.path.join(
    #                             labels_path, f"{filename}_{index}.txt"), "w"
    #                     ) as f:
    #                         new_w = float(w) / f_w
    #                         new_h = float(h) / f_h
    #                         new_x = 0.5
    #                         new_y = 0.5

    #                         # print(f"{name} {new_x} {new_y} {new_w} {new_h}")
    #                         f.write(f"{name} {new_x} {new_y} {new_w} {new_h}")

    #                     index += 1
    #             except Exception:
    #                 im = im.resize((640, 640))
    #                 im.save(img)
    #                 print("Failed to crop", img)
    #                 continue

    #             os.remove(img)
    #             os.remove(label)
