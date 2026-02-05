import collections
import os
import shutil
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import matplotlib

# matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO
import yaml


class Dataset:

    def __init__(self, name="dataset", path="./dataset", n_classes=0, class_map={}):
        self.name = name
        self.path = path
        self.n_classes = n_classes
        self.class_map = class_map
        self.loaded = False

        self.train_df = pd.DataFrame()
        self.val_df = pd.DataFrame()
        self.test_df = pd.DataFrame()

        self.images_path = os.path.join(path, "images")
        self.labels_path = os.path.join(path, "labels")

        if os.path.exists(path):
            shutil.rmtree(path)

        os.mkdir(path)
        os.mkdir(self.images_path)
        os.mkdir(self.labels_path)

    def delete(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    def from_raw(
        self,
        raw_path,
        preprocess,
        fraction=1,
        skew=False,
        normalize=False,
        max=0,
        equal=False,
    ):
        if preprocess is None:
            raise Exception("Preprocess not defined")

        preprocess(
            fig_name=self.name,
            src=raw_path,
            out=self.path,
            n_classes=self.n_classes,
            fraction=fraction,
            skew=skew,
            normalize=normalize,
            max=max,
            equal=equal,
        )

        self.loaded = True

    def from_datasets(self, datasets=[]):
        for dataset in datasets:
            if dataset.loaded:
                self.train_df = pd.concat(
                    [self.train_df, dataset.train_df], ignore_index=True
                )
                self.val_df = pd.concat(
                    [self.val_df, dataset.val_df], ignore_index=True
                )
                self.test_df = pd.concat(
                    [self.test_df, dataset.test_df], ignore_index=True
                )

                shutil.copytree(
                    dataset.images_path, self.images_path, dirs_exist_ok=True
                )
                shutil.copytree(
                    dataset.labels_path, self.labels_path, dirs_exist_ok=True
                )

        self.loaded = True

    def show_distribution(self, title, class_list, df):
        class_distribution = []
        class_labels = []

        color = plt.cm.viridis(np.linspace(0, 1, len(class_list)))
        for i in sorted(class_list):
            try:
                # class_distribution.append(100 * np.count_nonzero(df[i]) / df.shape[0])
                class_distribution.append(np.sum(df[i]))
                class_labels.append(self.class_map[int(i)])
            except Exception:
                class_distribution.append(0)

        # Make plot
        plt.figure(figsize=(10, 5))
        plt.bar(class_labels, class_distribution, color=color, width=1, edgecolor="k")
        plt.title(f"{title}    nImages: {str(len(df))}")
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
        plt.savefig(self.name + title, bbox_inches="tight")
        plt.close()

    def get_distribution(self):
        labels_column = []
        total_labels = []
        image_names_column = []

        # Get the labels from the txt files and make a label column list of all image labels (without duplicates)
        for split in ["train", "test", "val"]:
            for file in os.listdir(os.path.join(self.labels_path, split)):
                if os.path.isfile(os.path.join(self.labels_path, split, file)):
                    filename, extension = os.path.splitext(file)
                    image_names_column.append(filename + ".jpg")
                    with open(os.path.join(self.labels_path, split, file)) as txt:
                        orig_labels = []
                        for line in txt:
                            words = line.split()
                            orig_labels.append(int(words[0]))
                            total_labels.append(int(words[0]))
                        labels_column.append(orig_labels)

        # Create csv data
        csv_rows = zip(image_names_column, labels_column)

        # Load dataframe and convert to column per label
        df = pd.DataFrame(csv_rows, columns=["images", "labels"])
        labels = df["labels"]
        cs = collections.Counter(total_labels)

        text_to_category = {label: [] for label in cs.keys()}
        for _, item in df.iterrows():
            for label in text_to_category:
                text_to_category[label].append(item["labels"].count(label))

        for label in text_to_category:
            df[label] = text_to_category[label]

        del df["labels"]

        print(df.head(5))

        class_list = list(filter(lambda x: x != "images", df.columns))

        self.show_distribution(self.name, class_list, df)

    def generate_split_df(self, train, val, test, debug=False):

        labels_column = []
        total_labels = []
        image_names_column = []

        # Get the labels from the txt files and make a label column list of all image labels (without duplicates)
        for file in os.listdir(self.labels_path):
            if os.path.isfile(os.path.join(self.labels_path, file)):
                filename, extension = os.path.splitext(file)
                image_names_column.append(filename + ".jpg")
                with open(os.path.join(self.labels_path, file)) as txt:
                    orig_labels = []
                    for line in txt:
                        words = line.split()
                        orig_labels.append(int(words[0]))
                        total_labels.append(int(words[0]))
                    labels_column.append(orig_labels)

        # Create csv data
        csv_rows = zip(image_names_column, labels_column)

        # Load dataframe and convert to column per label
        df = pd.DataFrame(csv_rows, columns=["images", "labels"])
        labels = df["labels"]
        cs = collections.Counter(total_labels)

        text_to_category = {label: [] for label in cs.keys()}
        for _, item in df.iterrows():
            for label in text_to_category:
                text_to_category[label].append(item["labels"].count(label))

        for label in text_to_category:
            df[label] = text_to_category[label]

        del df["labels"]

        if debug:
            print(df.head(5))

        class_list = list(filter(lambda x: x != "images", df.columns))

        if debug:
            self.show_distribution("Before Split", class_list, df)

        X = labels.to_numpy()
        Y = df[class_list].to_numpy(dtype=np.float32)
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=1 - train, random_state=0
        )

        for train_index, tmp_index in msss.split(X, Y):
            train_list = train_index.tolist()
            tmp_list = tmp_index.tolist()
            for i in range(len(train_list)):
                train_list[i] = image_names_column[train_list[i]]

            for i in range(len(tmp_list)):
                tmp_list[i] = image_names_column[tmp_list[i]]

        train_df = df[df["images"].isin(train_list)]
        tmp_df = df[df["images"].isin(tmp_list)]
        tmp_index = df.index[df["images"].isin(tmp_list)].to_list()

        if test == 0:
            return train_df, tmp_df, pd.DataFrame()

        if val == 0:
            return train_df, pd.DataFrame(), tmp_df

        X = labels[tmp_index].to_numpy()
        Y = tmp_df[class_list].to_numpy(dtype=np.float32)
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=val / (test + val), random_state=0
        )

        for test_index, val_index in msss.split(X, Y):
            val_list = val_index.tolist()
            test_list = test_index.tolist()
            for i in range(len(val_list)):
                val_list[i] = tmp_list[val_list[i]]

            for i in range(len(test_list)):
                test_list[i] = tmp_list[test_list[i]]

        val_df = df[df["images"].isin(val_list)]
        test_df = df[df["images"].isin(test_list)]

        if debug:
            self.show_distribution("Train Split", class_list, train_df)
            self.show_distribution("Val Split", class_list, val_df)
            self.show_distribution("Test Split", class_list, test_df)

        return train_df, val_df, test_df

    def mv_img_to_split(self, split, df):
        images_path = os.path.join(self.images_path, split)
        labels_path = os.path.join(self.labels_path, split)

        if os.path.exists(images_path):
            shutil.rmtree(images_path)

        if os.path.exists(labels_path):
            shutil.rmtree(labels_path)

        os.mkdir(images_path)
        os.mkdir(labels_path)

        if df.empty:
            return

        for file in df["images"]:
            label = file.replace(".jpg", ".txt")

            og_path = os.path.join(self.images_path, file)
            target_path = os.path.join(images_path, file)
            os.replace(og_path, target_path)

            og_txt_path = os.path.join(self.labels_path, label)
            target_txt_path = os.path.join(labels_path, label)
            os.replace(og_txt_path, target_txt_path)

    def split(self, train=0.70, val=0.10, test=0.20):
        if not self.loaded:
            raise Exception("Dataset is not loaded")

        self.train_df, self.val_df, self.test_df = self.generate_split_df(
            train, val, test, True
        )

        self.mv_img_to_split("train", self.train_df)
        self.mv_img_to_split("val", self.val_df)
        self.mv_img_to_split("test", self.test_df)

    def add_weighted_imgs(self, split, dt, df):
        images_path = os.path.join(self.images_path, split)
        labels_path = os.path.join(self.labels_path, split)

        if df.empty:
            return

        for file in df["images"]:
            label = file.replace(".jpg", ".txt")

            og_path = os.path.join(dt.images_path, split, file)
            target_path = os.path.join(images_path, file)
            os.replace(og_path, target_path)

            og_txt_path = os.path.join(dt.labels_path, split, label)
            target_txt_path = os.path.join(labels_path, label)
            os.replace(og_txt_path, target_txt_path)

    def add_dt_weighted(self, dt, weight):
        train_imgs = int(self.train_df.shape[0] * weight)
        val_imgs = int(self.val_df.shape[0] * weight)
        test_imgs = int(self.test_df.shape[0] * weight)

        if not dt.loaded:
            raise Exception("Dataset is not loaded")

        train_imgs = min(train_imgs, dt.train_df.shape[0])
        val_imgs = min(val_imgs, dt.val_df.shape[0])
        test_imgs = min(test_imgs, dt.test_df.shape[0])

        add_train_df = dt.train_df.sample(n=train_imgs, random_state=1)
        add_val_df = dt.val_df.sample(n=val_imgs, random_state=1)
        add_test_df = dt.test_df.sample(n=test_imgs, random_state=1)

        self.add_weighted_imgs("train", dt, add_train_df)
        self.add_weighted_imgs("val", dt, add_val_df)
        self.add_weighted_imgs("test", dt, add_test_df)

    def train(self, labeled_classes, out, base_model="yolo11s.pt", aug=False):
        yaml_path = os.path.join(self.path, f"{self.name}.yaml")

        # Data to be written to the YAML file
        data = {
            "path": self.path,
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": labeled_classes,
        }

        # Writing the data to a YAML file
        with open(yaml_path, "w") as file:
            yaml.dump(data, file)

        model = YOLO(base_model)
        if not aug:
            model.train(
                data=yaml_path,
                epochs=200,
                patience=100,
                lr0=0.001,
                lrf=0.001,
                batch=8,
                hsv_h=0,
                hsv_s=0,
                hsv_v=0,
                translate=0,
                scale=0,
                fliplr=0,
                mosaic=0,
                erasing=0,
                project=out,
                device=-1,
                multi_scale=True,
                name=f"{self.name}_noAug",
            )
            model.val(
                data=yaml_path, split="test", plots=True, name=f"{self.name}_noAug_Test"
            )
        else:
            model.train(
                data=yaml_path,
                epochs=200,
                patience=100,
                lr0=0.001,
                lrf=0.001,
                batch=8,
                project=out,
                device=-1,
                multi_scale=True,
                name=self.name,
            )
            model.val(
                data=yaml_path, split="test", plots=True, name=f"{self.name}_Test"
            )
        model.export(format="torchscript")
