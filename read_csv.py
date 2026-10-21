
import csv

with open("products.csv", "r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        print(row)
