import pandas as pd

df1 = pd.read_csv('./Tripcom/Botanic Garden.csv', encoding='utf-8')
df1['Attraction'] = 'Botanic Garden'
df2 = pd.read_csv('./Tripcom/Garden by the bay.csv', encoding='utf-8')
df2['Attraction'] = 'Garden by the bay'
df3 = pd.read_csv('./Tripcom/Marina Bay Sands.csv', encoding='utf-8')
df3['Attraction'] = 'Marina Bay Sands'
df4 = pd.read_csv('./Tripcom/River Wonder.csv', encoding='utf-8')
df4['Attraction'] = 'River Wonder'
df5 = pd.read_csv('./Tripcom/Safari.csv', encoding='utf-8')
df5['Attraction'] = 'Safari'
df6 = pd.read_csv('./Tripcom/SEA Aquarium.csv', encoding='utf-8')
df6['Attraction'] = 'SEA Aquarium'
df7 = pd.read_csv('./Tripcom/Sentosa.csv', encoding='utf-8')
df7['Attraction'] = 'Sentosa'
df8 = pd.read_csv('./Tripcom/Singapore Flyer.csv', encoding='utf-8')
df8['Attraction'] = 'Singapore Flyer'
df9 = pd.read_csv('./Tripcom/Singapore Zoo.csv', encoding='utf-8')
df9['Attraction'] = 'Singapore Zoo'
df10 = pd.read_csv('./Tripcom/Skypark.csv', encoding='utf-8')
df10['Attraction'] = 'Skypark'
df11 = pd.read_csv('./Tripcom/USS.csv', encoding='utf-8')
df11['Attraction'] = 'USS'

merged_df = pd.concat([df1, df2, df3,df4,df5,df6,df7,df8,df9,df10,df11], ignore_index=True)

print(f"Number of rows in merged_df: {len(merged_df)}")

merged_df.to_csv("Tripcom_reviews_details.csv", index=False, encoding="utf-8")

