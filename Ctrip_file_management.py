import pandas as pd

df1 = pd.read_csv('./ctrip/Botanic Garden-1708232703.csv')
df1['Attraction'] = 'Botanic Garden'
df2 = pd.read_csv('./ctrip/Garden by the bay-1708232324.csv')
df2['Attraction'] = 'Garden by the bay'
df3 = pd.read_csv('./ctrip/Marina Bay Sands-1708232339.csv')
df3['Attraction'] = 'Marina Bay Sands'
df4 = pd.read_csv('./ctrip/River Wonder-1708232503.csv')
df4['Attraction'] = 'River Wonder'
df5 = pd.read_csv('./ctrip/Safari-1708232242.csv')
df5['Attraction'] = 'Safari'
df6 = pd.read_csv('./ctrip/SEA Aquarium-1708232453.csv')
df6['Attraction'] = 'SEA Aquarium'
df7 = pd.read_csv('./ctrip/Sentosa-1708232055.csv')
df7['Attraction'] = 'Sentosa'
df8 = pd.read_csv('./ctrip/Singapore Flyer-1708232682.csv')
df8['Attraction'] = 'Singapore Flyer'
df9 = pd.read_csv('./ctrip/Singapore Zoo-1708232589.csv')
df9['Attraction'] = 'Singapore Zoo'
df10 = pd.read_csv('./ctrip/Skypark-1708232368.csv')
df10['Attraction'] = 'Skypark'
df11 = pd.read_csv('./ctrip/USS-1708232159.csv')
df11['Attraction'] = 'USS'


merged_df = pd.concat([df1, df2, df3,df4,df5,df6,df7,df8,df9,df10,df11], ignore_index=True)

print(f"Number of rows in merged_df: {len(merged_df)}")

merged_df.to_csv("Ctrip_reviews_details.csv", index=False, encoding="utf-8")

