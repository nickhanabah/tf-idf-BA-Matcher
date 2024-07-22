import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tokenizers import trainers, Tokenizer, models

from datasets import Dataset
from tqdm.auto import tqdm
from openai import OpenAI

import urllib3
from typing import List
from datetime import datetime
import os 

class Data: 
    def __init__(self, api_key: str, training_df_size: int = 10): 
        self.client = OpenAI(api_key=api_key)
        self.nr = training_df_size

    def generate_training_data(self, nr: int, system_prompt: str, user_prompt:str): 
        data_list = []
        for i in range(nr): 
            completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            )
            data_list.append(completion.choices[0].message.content)
            if ((i + 1) % 10) == 0: 
                print(f"{i + 1} out {nr} done!")
        
        self.data_list = data_list
        current_datetime = datetime.now()
        datetime_string = current_datetime.strftime("%Y_%m_%d_%H_%M")
        pd.DataFrame(self.data_list, columns = ['JobAds']).to_csv(f'/workspaces/tf-idf-BA-Matcher/training_jobads/jobs_{datetime_string}.csv')
        return data_list 

    def retrieve_target_classes(self, link: str, sheet_page: str): 
        http = urllib3.PoolManager()
        response = http.request('GET', link)
        content = response.data
        xlsx = pd.ExcelFile(content)
        
        self.classes_df = pd.read_excel(xlsx, sheet_page)
        return pd.read_excel(xlsx, sheet_page)

    def generate_training_jobads(self, 
                                system_prompt: str = "Du bist ein Assistent im Recruiting zur Verfassung von Stellenanzeigen.", 
                                user_prompt: str = "Schreibe mir eine Stellenenzeige mit 200-500 Wörtern für einen echten Job. Achte darauf das du blue und white collar Jobs nutzt!"):
        job_list = self.generate_training_data(nr = self.nr, 
                                        system_prompt=system_prompt, 
                                        user_prompt=user_prompt
                                        )
        jobs_df = pd.DataFrame(job_list)
        jobs_df.columns = ['Stellenanzeigen']
        self.jobs_df = jobs_df

    def retrieve_and_process_target_classes(self): 
        ba_df = self.retrieve_target_classes(
        link = 'https://statistik.arbeitsagentur.de/DE/Statischer-Content/Grundlagen/Klassifikationen/Klassifikation-der-Berufe/KldB2010-Fassung2020/Systematik-Verzeichnisse/Generische-Publikationen/Alphabetisches-Verzeichnis-Berufsbenennungen.xlsx?__blob=publicationFile&v=8+', 
        sheet_page = 'alphabet_Verz_Berufsb'
        )

        ba_df.columns = ba_df.iloc[3,:]
        ba_df = ba_df.iloc[4:,:]
        ba_df['KldB 2010 (5-Steller)'] = pd.to_numeric(ba_df['KldB 2010 (5-Steller)'], errors='coerce')
        ba_df = ba_df.loc[:, :].dropna()
        ba_df['KldB 2010 (5-Steller)'] = ba_df['KldB 2010 (5-Steller)'].astype(int)
        self.ba_df = ba_df.dropna(how='all')

    def list_files(self, directory):
        entries = os.listdir(directory)
        files = [entry for entry in entries if os.path.isfile(os.path.join(directory, entry))]
        return files

    def load_all_job_ads(self, directory): 
        files = self.list_files(directory)
        loading_paths = [f"{directory}/{file}" for file in files]
        job_list = []
        for path in loading_paths: 
            jobs = pd.read_csv(path)
            job_list = job_list + jobs['JobAds'].tolist()
        return job_list
        
class Classifier: 
    def __init__(self, data_object: Data, train: bool = False, load_job_ads: bool = True): 

        self.data_object = data_object
        if load_job_ads: 
            job_list = self.data_object.load_all_job_ads('/workspaces/tf-idf-BA-Matcher/training_jobads/')
            self.train_target_col = 'Stellenanzeigen'
            self.train_df = pd.DataFrame(job_list, columns=[self.train_target_col])
        else: 
            self.data_object.generate_training_jobads()
            self.train_df = self.data_object.jobs_df
            self.train_target_col = self.data_object.jobs_df.columns[0]

        self.data_object.retrieve_and_process_target_classes()
        self.classes_df = self.data_object.ba_df
        if train: 
            self.training()

    def tokenize_text(self, df: pd.DataFrame, target_columns: str): 
        assert self.tokenizer is not None
        return [self.tokenizer.encode(text).tokens for text in tqdm(df[target_columns].tolist())]

    def train_corp_iter(self,
                        iter_steps: int = 1000): 
        for i in range(0, len(self.train_df), iter_steps):
            yield self.train_df[i : i + iter_steps][self.train_target_col]

    def train_tokenizer(self, 
                        iter_steps: int = 1000, 
                        vocab_size: int = 2500, 
                        unk_token: str = "[UNK]", 
                        special_tokens: List[str] = ["[UNK]", "[SEP]"] 
                        ): 
        tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        
        dataset = Dataset.from_pandas(self.train_df[[self.train_target_col]])
        tokenizer.train_from_iterator(self.train_corp_iter(iter_steps), trainer=trainer)
        return tokenizer

    def identity_function(self, t):
        return t

    def train_vectorizer(self, tokenized_df_text, ngram_range: tuple=(1, 5), lowercase: bool = False):
        vec = TfidfVectorizer(ngram_range=ngram_range, lowercase=lowercase, sublinear_tf=True, analyzer='word',
                                tokenizer=self.identity_function, preprocessor=self.identity_function, token_pattern=None, strip_accents='unicode')
        vec.fit_transform(tokenized_df_text)
        return vec

    def training(self, vocab_size: int = 2500): 
        self.tokenizer = self.train_tokenizer(vocab_size = vocab_size)
        tokenized_texts_train = self.tokenize_text(self.train_df, self.train_target_col)
        self.tokenized_classes = self.tokenize_text(self.classes_df, 'Berufsbenennungen')
        self.vec = self.train_vectorizer(tokenized_texts_train)

    def predict_top_n_classes(self, target_text: str, top_n: int = 5): 
        assert self.tokenizer is not None
        assert self.tokenized_classes is not None
        assert self.vec is not None

        tokenized_target_text= self.tokenizer.encode(target_text).tokens
        vectorized_target_text = self.vec.transform([tokenized_target_text])
        
        try:
            self.vectorized_classes
            if self.vectorized_classes.shape[1] != vectorized_target_text.shape[1]:   
                self.vectorized_classes = self.vec.transform(self.tokenized_classes)
        except:  
            self.vectorized_classes = self.vec.transform(self.tokenized_classes)  
            
        similarity_index_list = cosine_similarity(vectorized_target_text, self.vectorized_classes).flatten()
        idx = similarity_index_list.argsort()[-top_n:].tolist()[::-1]
        result_df = self.classes_df.iloc[idx,:]
        return result_df.assign(cosine_similarities=similarity_index_list[idx].tolist())