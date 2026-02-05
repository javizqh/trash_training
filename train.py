#!/usr/bin/env python3


from extra.dataset import Dataset
from extra.preprocess import Preprocess

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

red_classes = {
    0: "Aluminium foil",
    1: "Bottle",
    2: "Bottle cap",
    3: "Can",
    4: "Paper/Cardboard",
    5: "Cup",
    6: "Food waste",
    7: "Glass",
    8: "Plastic",
    9: "Textile",
    10: "Scrap metal",
    11: "Styrofoam piece",
    12: "Unlabeled litter",
    13: "Cigarette",
}

red_classes_map = [
    0,
    12,
    12,
    1,
    2,
    7,
    3,
    4,
    5,
    6,
    7,
    8,
    8,
    4,
    4,
    8,
    8,
    8,
    8,
    10,
    9,
    10,
    9,
    12,
    12,
    11,
    12,
    13,
]


ultra_red_classes = {
    0: "Aluminium foil",
    1: "Bottle",
    2: "Bottle cap",
    3: "Can",
    4: "Paper/Cardboard",
    5: "Cup",
    6: "Food waste",
    7: "Glass",
    8: "Plastic",
}

ultra_red_classes_map = [
    0,
    -1,
    -1,
    1,
    2,
    7,
    3,
    4,
    5,
    6,
    7,
    8,
    8,
    4,
    4,
    8,
    8,
    8,
    8,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
]

n_150_classes = {
    0: "Bottle",
    1: "Bottle cap",
    2: "Can",
    3: "Carton",
    4: "Cup",
    5: "Other Plastic",
    6: "Paper",
    7: "Plastic bag & wrapper",
    8: "Straw",
    9: "Styrofoam piece",
    10: "Unlabeled litter",
    11: "Cigarette",
}

n_150_classes_map = [
    -1,
    -1,
    -1,
    0,
    1,
    -1,
    2,
    3,
    4,
    -1,
    -1,
    -1,
    5,
    6,
    -1,
    7,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    8,
    9,
    10,
    11,
]


def load_real_dt(prep):
    real_dts = []

    taco = Dataset("TACO", "./datasets/TACO", 28, orig_classes)
    taco.from_raw("./datasets/TACOraw", prep.mantain, skew=True, normalize=True)
    taco.split(val=0.15, test=0.15)
    taco.from_raw("./datasets/TACO", prep.r_crop_splited, equal=True)
    taco.train(orig_classes, "New-Results", base_model="yolo11s.pt")
    real_dts.append(taco)

    ext_taco = Dataset("ExtendedTACO", "./datasets/ExtendedTACO", 28, orig_classes)
    ext_taco.from_raw(
        "./datasets/ExtendedTACOraw", prep.mantain, skew=True, normalize=True
    )
    ext_taco.split(val=0.15, test=0.15)
    ext_taco.from_raw("./datasets/ExtendedTACO", prep.r_crop_splited, equal=True)
    ext_taco.train(orig_classes, "New-Results", base_model="yolo11s.pt", aug=True)
    # real_dts.append(ext_taco)

    uav = Dataset("UAVWaste", "./datasets/UAVWaste", 28, orig_classes)
    uav.from_raw("./datasets/UAVWasteraw", prep.mantain)
    uav.split()
    uav.from_raw("./datasets/UAVWaste", prep.r_crop_splited, equal=True)
    uav.train(orig_classes, "New-Results", base_model="yolo11s.pt", aug=True)
    real_dts.append(uav)

    return real_dts


if __name__ == "__main__":
    prep = Preprocess(n_classes=len(orig_classes), class_map=orig_classes)

    taco = Dataset("miniTACO", "./datasets/miniTACO", 28, orig_classes)
    taco.from_raw("./datasets/miniTACOraw", prep.mantain, skew=True, normalize=True)
    taco.split(val=0.15, test=0.15)
    taco.from_raw("./datasets/miniTACO", prep.r_crop_splited)
    taco.train(orig_classes, "New-Results-Mini", base_model="yolo11s.pt", aug=False)
    taco.train(orig_classes, "New-Results-Mini", base_model="yolo11s.pt", aug=True)
