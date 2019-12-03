from utils import load_json, save_json


def filter_data(file_name):
    filtered_data = []
    raw_data = load_json(file_name)
    for ex in raw_data:
        if not any(x in ex['vid_name'] for x in ["friends", "met", "grey", "house", "castle"]):
            filtered_data.append(ex)
    save_json(filtered_data, file_name)


def main():
    train_path = "data/tvqa_train_processed.json"
    valid_path = "data/tvqa_val_processed.json"
    test_path = "data/tvqa_test_public_processed.json"

    filter_data(train_path)
    filter_data(valid_path)
    filter_data(test_path)


if __name__ == "__main__":
    main()
