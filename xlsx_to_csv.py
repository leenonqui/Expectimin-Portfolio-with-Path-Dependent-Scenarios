import pandas as pd

# Load the Excel file
file_path = "JSTdatasetR6.xlsx"
xls = pd.ExcelFile(file_path)

df = pd.read_excel(xls)

usa_df = df[df['iso'] == 'USA']

column_list = []
for column in usa_df.columns:
    if not (column == 'gdp' or column == 'cpi' or column == 'year'):
        column_list.append(column)
# usa_df.set_index('year')
print(f"Columns to be dropped: \n{column_list}'\n\nNumber of Columns to drop: {len(column_list)}.")
usa_df = usa_df.drop(columns = column_list)

print(usa_df.head())

# Save a csv file of the data needed
usa_df.to_csv(r"~/workspace/github.com/leenonqui/Expectiminimax-Portfolio-with-Path-Dependent-Scenarios/usa_gdp_cpi.csv")

