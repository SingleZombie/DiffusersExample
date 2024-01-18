from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="dataset/mountain")
print(dataset)
print(dataset["train"].column_names)
print(dataset["train"]['image'])
print(dataset["train"]['text'])