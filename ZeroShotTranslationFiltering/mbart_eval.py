from datasets import load_dataset
import pandas as pd
import nltk
from tqdm import tqdm

dataset_itihasa = load_dataset("rahular/itihasa")
dataset_iitb = load_dataset("cfilt/iitb-english-hindi")

dataset_itihasa.set_format(type='pd')
dataset_iitb.set_format(type='pd')

df_train1 = pd.DataFrame(dataset_itihasa['train']['translation'].to_list())[['sn','en']]
df_valid1 = pd.DataFrame(dataset_itihasa['validation']['translation'].to_list())[['sn','en']]
df_test1 = pd.DataFrame(dataset_itihasa['test']['translation'].to_list())[['sn','en']]

# df_train1['sn'] = df_train1['sn'].apply(lambda x: '[2EN] '+x)
# df_valid1['sn'] = df_valid1['sn'].apply(lambda x: '[2EN] '+x)
# df_test1['sn'] = df_test1['sn'].apply(lambda x: '[2EN] '+x)

df_train2 = pd.DataFrame(dataset_iitb['train']['translation'].to_list())[['en','hi']]
df_valid2 = pd.DataFrame(dataset_iitb['validation']['translation'].to_list())[['en','hi']]
df_test2 = pd.DataFrame(dataset_iitb['test']['translation'].to_list())[['en','hi']]

# df_train2['en'] = df_train2['en'].apply(lambda x: '[2HI] '+x)
# df_valid2['en'] = df_valid2['en'].apply(lambda x: '[2HI] '+x)
# df_test2['en'] = df_test2['en'].apply(lambda x: '[2HI] '+x)

df_train1.columns = ['inp_lang', 'tar_lang']
df_train2.columns = ['inp_lang', 'tar_lang']

df_valid1.columns = ['inp_lang', 'tar_lang']
df_valid2.columns = ['inp_lang', 'tar_lang']

df_test1.columns = ['inp_lang', 'tar_lang']
df_test2.columns = ['inp_lang', 'tar_lang']

print(len(df_train2), len(df_valid2), len(df_test2))

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

bl_1, bl_2, bl_4 = 0,0,0
for row in tqdm(df_test2.itertuples(), total=df_test2.shape[0]):
  # print(row)
  print(row[0], row[1], row[2]) # ind, eng, hi
  # translate Hindi to Eng
  tokenizer.src_lang = "hi_IN"
  encoded_hi = tokenizer(row[2], return_tensors="pt")
  generated_tokens = model.generate(
      **encoded_hi,
      forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
  )
  pred = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
  # print(row[1], pred)
  res = nltk.translate.bleu_score.sentence_bleu([row[1]], pred, weights=[(1.0,), (0.5, 0.5), (0.25, 0.25, 0.25, 0.25)]) # ref is english
  # print(res)
  bl_1 += res[0]
  bl_2 += res[1]
  bl_4 += res[2]

print("Results: ")
print(bl_1/len(df_test2), bl_2/len(df_test2), bl_4/len(df_test2))

bl_1, bl_2, bl_4 = 0,0,0
for row in df_test2.itertuples():
  # print(row)
  print(row[0], row[1], row[2]) # ind, eng, hi
  # translate Eng to Hindi
  tokenizer.src_lang = "en_XX"
  encoded_hi = tokenizer(row[1], return_tensors="pt")
  generated_tokens = model.generate(
      **encoded_hi,
      forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
  )
  pred = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
  # print(row[1], pred)
  res = nltk.translate.bleu_score.sentence_bleu([row[2]], pred, weights=[(1.0,), (0.5, 0.5), (0.25, 0.25, 0.25, 0.25)]) # ref is hindi
  # print(res)
  bl_1 += res[0]
  bl_2 += res[1]
  bl_4 += res[2]

print("Results: ")
print(bl_1/len(df_test2), bl_2/len(df_test2), bl_4/len(df_test2))
