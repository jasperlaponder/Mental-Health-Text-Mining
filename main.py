from import_data import load_data, split_data

if __name__ == "__main__":
    data = load_data(True)
    pre_covid, post_covid = split_data(data)
    print(pre_covid)
    print(post_covid)