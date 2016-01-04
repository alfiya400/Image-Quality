__author__ = 'alfiya'
import os
import json
import pandas as pd


def get_img_files(folderpath, img_ext=(".jpg", ".png", ".bmp", ".jpeg")):
    # iter over images
    img_list = []
    for f in os.listdir(folderpath):
        img_path = os.path.join(folderpath,f)
        filename, file_extension = os.path.splitext(img_path)
        if os.path.isfile(img_path) and file_extension in img_ext:
            img_list.append(img_path)

    return img_list


def load_data(filepath, folderpath, distortions_mapping_path):
    def get_distortion_type(x, mapping):
        t = x.split("_")[1]
        l = int(x.split("_")[-1].split(".")[0])
        return pd.Series({"type": t, "type_name": mapping.get(t), "level": l})

    mapping = json.load(open(distortions_mapping_path))
    df = pd.read_table(filepath, sep=" ", header=None,
                       names=["mos", "path"],
                       dtype=dict(mos=float, path=str))

    if pd.isnull(df).sum().sum() > 0:
        raise ValueError("NULL values in a dataset :(")
    X = df["path"].apply(lambda x: os.path.join(folderpath, x)).values
    y = df["mos"].values
    distortions = df["path"].apply(get_distortion_type, mapping=mapping)
    return X, y, distortions



