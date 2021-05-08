import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--foot-train', default='./data/annotations/person_keypoints_train2017_foot_v1.json')
    parser.add_argument('--foot-val', default='./data/annotations/person_keypoints_val2017_foot_v1.json')
    args = parser.parse_args()

    with open(args.foot_train, "r") as f:
        dataset = json.load(f)
    dataset['categories'] = [dataset['categories']]
    with open(args.foot_train, "w") as f:
        json.dump(dataset, f)


    with open(args.foot_val, "r") as f:
        dataset = json.load(f)
    dataset['categories'] = [dataset['categories']]
    with open(args.foot_val, "w") as f:
        json.dump(dataset, f)