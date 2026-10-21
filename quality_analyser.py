import csv

def load_products_from_csv(filepath):  
  products = []
  try:
    with open(filepath, "r", newline="") as file:
      reader = csv.DictReader(file)
      for row in reader:
        clean_item = {
            "name": row["name"],
            "defective": row["defective"] == "True"
        }
        products.append(clean_item)
  except FileNotFoundError:
      print(f"Error: The file {filepath} was not found.")
      return []

  return products


def analyse_quality(products):
    total = len(products)
    if total == 0:
        print("No products to analyse")
        return None #stops function execution right there 
    defective_count = 0
    defective_names = []

    for product in products:
        if product["defective"]:
            defective_count += 1
            defective_names.append(product["name"])

    defect_rate = (defective_count / total) * 100


    print("Total products:", total)
    print("Defective products:", defective_count)
    print("Defect rate:", defect_rate, "%")
    print("Defective product names:", defective_names)

    if defect_rate >25.0:
        print("STATUS: Quality Check Failed - Action Required")
    else:
        print("STATUS: Quality Check Passed")

    return{
        "total": total,
        "defective": defective_count,
        "defect_rate": defect_rate,
        "defective_names":defective_names
            
    }
def save_report_to_csv(report, filepath):
    if report is None:
        print("No report data to save.")
        return

    fieldnames = ["total", "defective", "defect_rate"]

    with open(filepath, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "total": report["total"],
            "defective": report["defective"],
            "defect_rate": report["defect_rate"]
        })

    print(f"Report successfully saved to {filepath}")



empty_report=analyse_quality([])
print("Empty list report:", empty_report)

print("\n___ Testing file not found error___")
missing_data = load_products_from_csv("non_existent_file.csv")
analyse_quality(missing_data)

print("___Testing valid CSV___")
products = load_products_from_csv("products.csv")
report = analyse_quality(products)
save_report_to_csv (report, "quality_report.csv")
  