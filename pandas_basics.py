import pandas as pd 
df = pd.read_csv("products.csv")
print (df.shape)
defective_products = df[df["defective"] == True]
print(defective_products)
total_count = df.shape[0]
defective_count = defective_products.shape[0]

print("Total products:", total_count)
print("Defective count:", defective_count)

defect_rate = (defective_count / total_count) * 100
print("Defect rate percentage:", defect_rate, "%")

