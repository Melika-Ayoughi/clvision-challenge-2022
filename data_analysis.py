import json
import numpy as np
import matplotlib.pyplot as plt
import os
import nltk
import pandas
from nltk.corpus import wordnet


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def hist():
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


def build_hierarchy(path="./hierarchy.json"):
    if os.path.exists(path):
        return load_json(path)

    train = load_json("../../../project/mayoughi/dataset/train_.json")
    name_to_synset = load_json("./name_to_synset.json")

    categories_without_synset = set()
    edges = set()
    for cat in train["categories"]:
        synset_name = "_".join(cat['name'].split(" "))

        # if len(wordnet.synsets(synset_name, pos=wordnet.NOUN)) == 0:
            # categories_without_synset.add(synset_name)

        if synset_name in name_to_synset:
            frequent_synset = wordnet.synset(name_to_synset[synset_name])
        else:
            # if synset_name not in categories_without_synset:
            # find most frequent synset
            freq = -1
            for synset in wordnet.synsets(synset_name, pos=wordnet.NOUN):
                sums = sum(lemma.count() for lemma in synset.lemmas())
                if sums > freq:
                    frequent_synset = synset
                    freq = sums
        father = None
        # choose the hypernym with the shortest path to root
        min_hypernym_path = min(frequent_synset.hypernym_paths(), key=len)
        if min_hypernym_path[1].name() != 'physical_entity.n.01':
            h_paths = [h_path for h_path in frequent_synset.hypernym_paths() if h_path[1].name() == 'physical_entity.n.01' ]
            if h_paths:
                min_hypernym_path = min(h_paths, key=len)

        if synset_name in ["Facial_tissue_holder", "XBox", "PlayStation", "Picnic_basket", "Nightstand", "Nightstand",
                           "Infant_bed", "High_heels", "Dog_bed", "Bathroom_cabinet", "Bicycle_helmet", "Table_tennis_racket"]:
            min_hypernym_path.append(f"{synset_name}.n.01")
        for child in min_hypernym_path:
            if father is None:
                father = child
                continue
            # print(f"{father}, {child}")
            if not isinstance(child, nltk.corpus.reader.wordnet.Synset):
                child_name = child
            else:
                child_name = child.name()
            if not isinstance(father, nltk.corpus.reader.wordnet.Synset):
                father_name = father
            else:
                father_name = father.name()

                edges.add((father_name, child_name))
            father = child
        # print(frequent_synset.hypernym_paths())
    nodes = {}
    root = next(iter(set(start for start, _ in edges) - set(end for _, end in edges)))
    for start, end in edges:
        nodes.setdefault(start, {})[end] = nodes.setdefault(end, {})

    tree = {root: nodes[root]}
    save_json(tree, path)

    contract(tree)
    save_json(tree, path)

    return tree


def contract(tree):
    to_check = list(tree.keys())
    next_to_check = []
    while to_check:
        for id in to_check:
            subtree = tree[id]
            if len(subtree) == 1:
                # remove the child
                del (tree[id])
                subsubid = list(subtree.keys())[0]
                subsubval = list(subtree.values())[0]
                tree[subsubid] = subsubval
                next_to_check.append(subsubid)
        to_check = next_to_check
        next_to_check = []
    for subtree in tree.values():
        contract(subtree)


def path_to_leaves(root, tree, path, pathLen, all_paths):
    if (len(path) > pathLen):
        path[pathLen] = root
        pathLen += 1

    elif root != "start":
        path.append(root)
        pathLen += 1

    if not tree.values():
        all_paths.append(path[0: pathLen])
        for i in path[0: pathLen]:
            print(i, " ", end="")
        print()
    else:
        for subid, subtree in tree.items():
            # print(path)
            path_to_leaves(subid, subtree, path, pathLen, all_paths)


def transitive_closure(all_paths):

    # make sure each edge is included only once
    edges = set()
    for path in all_paths:
        path.reverse()
        for child in range(len(path)):
            for parent in range(child + 1, len(path)):
                edges.add((path[child], path[parent]))

    nouns = pandas.DataFrame(list(edges), columns=['id1', 'id2'])
    nouns['weight'] = 1

    nouns.to_csv('./noun_closure.csv', index=False)
    return nouns


if __name__ == "__main__":
    train = load_json("../../../project/mayoughi/dataset/train_.json")
    tree = build_hierarchy("./contracted_hierarchy.json")
    # tree = {1: {2: {3: {}, 4: {}}, 5: {6: {}, 7: {}, 8: {}}, 9: {}}}
    path, all_paths = [], []
    path_to_leaves("start", tree, path, 0, all_paths)
    nouns = transitive_closure(all_paths)

    # sorted_ind = creat_histogram("../../../project/mayoughi/dataset/train_.json", "histogram_train")
    # creat_histogram("./test_.json", "histogram_test", sorted_ind)
    # creat_histogram("../../../project/mayoughi/dataset/validation_.json", "histogram_val", sorted_ind)
