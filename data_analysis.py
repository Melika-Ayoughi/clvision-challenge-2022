import json
import numpy as np
import matplotlib.pyplot as plt
import os

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def hist:
    train_original = load_json("../../../project/mayoughi/dataset/ego_objects_challenge_train_original.json")
    classes = [ann["instance_id"].split("_")[0] for ann in train_original["annotations"] if ann['id'] != 127172]

    class_to_instance = {}
    for cls in classes:
        if cls not in class_to_instance:
            class_to_instance[cls] = 1
        else:
            class_to_instance[cls] += 1

    fig = plt.figure(figsize=(10, 8))
    sorted_class_to_instance = sorted(class_to_instance.items(), key=lambda kv: kv[1], reverse=True)
    plt.bar(range(1110), height=[v for k, v in sorted_class_to_instance], color='#3DA4AB')
    # plt.xticks(bins[:-1], ind_sorted, rotation=90, fontsize=10)
    fig.savefig(os.path.join("", f"histogram_original.pdf"))
    plt.clf()


# classes = [ann["instance_id"].split("_")[0] for ann in train_original["annotations"]]
def creat_histogram(annotation_path, save_pdf_path, sort_order=None):
    annotations = load_json(annotation_path)
    classes = [ann["instance_id"].split("_")[0] for ann in annotations["annotations"] if ann['id'] != 127172]
    class_to_instance = {}
    for cls in classes:
        if cls not in class_to_instance:
            class_to_instance[cls] = 1
        else:
            class_to_instance[cls] += 1

    fig = plt.figure(figsize=(10, 8))

    sorted_class_to_instance = dict(sorted(class_to_instance.items(), key=lambda kv: kv[1], reverse=True))  # sort based on value
    if sort_order is None:
        plt.bar(range(1110), height=[v for k, v in sorted_class_to_instance.items()], color='#3DA4AB')
    else:
        plt.bar(range(1110), height=[sorted_class_to_instance[ind] for ind in sort_order], color='#3DA4AB')

    # plt.xticks(bins[:-1], ind_sorted, rotation=90, fontsize=10)
    fig.savefig(os.path.join("", f"{save_pdf_path}.pdf"))
    plt.clf()

    sorted_ind = [k for k in sorted_class_to_instance.keys()]
    return sorted_ind

if __name__ == "__main__":
    sorted_ind = creat_histogram("../../../project/mayoughi/dataset/train_.json", "histogram_train")
    creat_histogram("./test_.json", "histogram_test", sorted_ind)
    creat_histogram("../../../project/mayoughi/dataset/validation_.json", "histogram_val", sorted_ind)
