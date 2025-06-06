import pandas as pd

# Load the Excel file
file_path = "/mnt/data/JSTdatasetR6.xlsx"
xls = pd.ExcelFile(file_path)

df = xls.parse("Sheet1")

print(df.head())

