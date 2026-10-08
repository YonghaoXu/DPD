import os
import json
import shutil
import argparse
from tqdm import tqdm


def gen_metadata_caption(args):

    dataset_config = {
        "UCM": ("./dataset/UCM_train.txt", {
            "agricultural": "agricultural area", "airplane": "airplane",
            "baseballdiamond": "baseball diamond", "beach": "beach",
            "buildings": "buildings", "chaparral": "chaparral",
            "denseresidential": "dense residential area", "forest": "forest",
            "freeway": "freeway", "golfcourse": "golf course",
            "harbor": "harbor", "intersection": "intersection",
            "mediumresidential": "medium residential area", "mobilehomepark": "mobile home park",
            "overpass": "overpass", "parkinglot": "parking lot",
            "river": "river", "runway": "runway", "sparseresidential": "sparse residential area",
            "storagetanks": "storage tanks", "tenniscourt": "tennis court"
        }),
        "AID": ("./dataset/AID_train.txt", {
            "airport": "airport", "bareland": "bare land", "baseballfield": "baseball field",
            "beach": "beach", "bridge": "bridge", "center": "city center", "church": "church",
            "commercial": "commercial area", "denseresidential": "dense residential area",
            "desert": "desert", "farmland": "farmland", "forest": "forest",
            "industrial": "industrial area", "meadow": "meadow", "mediumresidential": "medium residential area",
            "mountain": "mountain", "park": "park", "parking": "parking lot", "playground": "playground",
            "pond": "pond", "port": "port", "railwaystation": "railway station", "resort": "resort",
            "river": "river", "school": "school", "sparseresidential": "sparse residential area",
            "square": "city square", "stadium": "stadium", "storagetanks": "storage tanks",
            "viaduct": "viaduct"
        }),
        "NWPU": ("./dataset/NWPU_train.txt", {
            "airplane": "airplane", "airport": "airport", "baseball_diamond": "baseball diamond",
            "basketball_court": "basketball court", "beach": "beach", "bridge": "bridge",
            "chaparral": "chaparral", "church": "church", "circular_farmland": "circular farmland",
            "cloud": "cloud", "commercial_area": "commercial area", "dense_residential": "dense residential area",
            "desert": "desert", "forest": "forest", "freeway": "freeway", "golf_course": "golf course",
            "ground_track_field": "ground track field", "harbor": "harbor", "industrial_area": "industrial area",
            "intersection": "intersection", "island": "island", "lake": "lake", "meadow": "meadow",
            "medium_residential": "medium residential area", "mobile_home_park": "mobile home park",
            "mountain": "mountain", "overpass": "overpass", "palace": "palace", "parking_lot": "parking lot",
            "railway": "railway", "railway_station": "railway station", "rectangular_farmland": "rectangular farmland",
            "river": "river", "roundabout": "roundabout", "runway": "runway", "sea_ice": "sea ice",
            "ship": "ship", "snowberg": "snowberg", "sparse_residential": "sparse residential area",
            "stadium": "stadium", "storage_tank": "storage tank", "tennis_court": "tennis court",
            "terrace": "terrace", "thermal_power_station": "thermal power station", "wetland": "wetland"
        })
    }

    train_file, class_name_mapping = dataset_config[args.dataset]
    dst_root = os.path.join(args.dst_root, args.dataset, "train")
    os.makedirs(dst_root, exist_ok=True)

    metadata = []

    with open(train_file, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f"Generating template captions for {args.dataset}"):

        path_label = line.strip().split()[0]
        src_img_path = os.path.join(args.src_root, path_label)

        dst_img_name = os.path.basename(src_img_path)
        dst_img_path = os.path.join(dst_root, dst_img_name)

        cls = os.path.basename(os.path.dirname(path_label)).lower()

        if cls not in class_name_mapping:
            print(f"Warning: class {cls} not in mapping. Skipped.")
            continue

        readable_name = class_name_mapping[cls]

        # copy image
        if not os.path.exists(dst_img_path):
            shutil.copy(src_img_path, dst_img_path)

        caption = f"A satellite image of {readable_name}"

        metadata.append({
            "file_name": dst_img_name,
            "text": caption
        })

    output_path = os.path.join(dst_root, "metadata.jsonl")
    with open(output_path, "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")

    print(f"Template metadata saved to {output_path}. Total images: {len(metadata)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="UCM")
    parser.add_argument("--src_root", type=str, default="/proj/cvl/users/x_xuyon/Data/VisionLanguage/")
    parser.add_argument("--dst_root", type=str, default="/proj/cvl/users/x_xuyon/Data/VisionLanguage/Caption/")
    args = parser.parse_args()
    gen_metadata_caption(args)