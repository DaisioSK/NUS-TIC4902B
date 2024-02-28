import pandas as pd
import langid
import regex as re
from opencc import OpenCC
from langdetect import detect


# Define default source path
SRC_PATH = "./src/"

# Load dataset
df_review = pd.read_csv(SRC_PATH + "Tripcom_reviews_details.csv")

print(f"Currently, review dataset has {len(df_review)} rows.")
print(df_review.head())

df_review.dropna()
print(f"Currently, review dataset has {len(df_review)} rows.")

# List key columns that are used to identify the duplicate rows
key_columns_trip = ["content"]

# Find out duplicated rows and keep first one
duplicated_rows_trip = df_review[df_review.duplicated(subset=key_columns_trip, keep="first")]

# Remove duplicated rows and keep first one
df_review = df_review.drop_duplicates(subset=key_columns_trip, keep="first").reset_index(drop=True)
print(f"Currently, review dataset has {len(df_review)} rows.")

df_review['time'] = pd.to_datetime(df_review['time'], format='%d/%m/%Y')

df_review = df_review[df_review['time'] >= pd.Timestamp('2022-04-01')]
print(f"Currently, review dataset has {len(df_review)} rows.")
df_review['time'] = df_review['time'].dt.strftime('%d/%m/%y')

# df_review.to_csv('new_data.csv', index=False, encoding='utf-8')

# Function to check whether text is English
def is_english(text):
    if pd.isna(text):
        return False  # if value is null, return False
    lang, _ = langid.classify(text)
    return lang == "en"

# Create a new column "is_english" to indicate whether the review is wrote in English
df_review.loc[:, "is_english"] = df_review["content"].apply(is_english)

# Keep English reviews and save into df_review_english dataset
df_review_english = df_review[df_review["is_english"]]
df_review_english = df_review_english.reset_index(drop=True)

df_review_english.head(2)

df_review_english.to_csv('eng_data.csv', index=False, encoding='utf-8')
def convert_to_simplified_chinese(text):
    # 判断每一列是否存在日文字符

    cc = OpenCC('t2s')
    return cc.convert(text)

def is_japanese(text):
    if pd.isna(text):
        return False  # if value is null, return False
    lang, _ = langid.classify(text)
    return lang == "ja"



chinese_pattern = re.compile(r'[\p{Han}]')
df_review['content'] = df_review['content'].apply(convert_to_simplified_chinese)

# Create a new column "is_english" to indicate whether the review is wrote in English
df_review.loc[:, "is_japanese"] = df_review["content"].apply(is_japanese)

# Keep English reviews and save into df_review_english dataset
df_review_no_japanese = df_review[df_review["is_japanese"]==False]
df_review_no_japanese = df_review_no_japanese.reset_index(drop=True)

df_review_chinese = df_review_no_japanese[df_review_no_japanese['content'].apply(lambda x: bool(chinese_pattern.search(x)))]
df_review_chinese.to_csv('Chinese.csv', index=False, encoding='utf-8')

print('successful')