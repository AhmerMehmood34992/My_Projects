import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenizer


# with open("counselchat-data.csv", "rb") as file:
#     result = chardet.detect(file.read())
#     print(result)
import pandas as pd
# d1=d1.rename(columns={'question':'Question','answer':'Answer'})


# with open("counselchat-data.csv", "rb") as file:
#     result = chardet.detect(file.read())
#     print(result)






d1 = pd.read_csv("cleaned_train.csv" )
df1=pd.DataFrame(d1)

# print(df1.head())

# d2=pd.read_csv("20200325_counsel_chat.csv" )
# df2=pd.DataFrame(d2)
# df2=df2.rename(columns={'questionTitle':'question','answerText':'answer'})

# df2=df2[['question','answer']]
# # print(df2.head())

# d3=pd.read_csv("counselchat-data.csv" )
# df3=pd.DataFrame(d3)

# df3=df3[['question','answer']]


# df=pd.concat([df1,df2,df3],axis=0)
# print(df)

# df.to_csv('Nlp_project_data.csv', index=False) 








df=pd.DataFrame(pd.read_csv('Nlp_project_data.csv'))


# regex
# import csv
# import re


# pattern = re.compile(r'[^A-Za-z0-9\s\.\?]+|http\S+')


# def clean_text(text):
#     text = re.sub(pattern, '', text)  # Remove unwanted characters
#     text = text.strip()              # Trim leading and trailing spaces
#     return text

# csv_files = [
#     {'input_file': 'Nlp_project_data.csv', 'output_file': 'cleaned_Nlp_project_data.csv'},
#     # {'input_file': 'counselchat-data.csv', 'output_file': 'cleaned_counselchat.csv'}
# ]

# # Process each CSV file
# for file_info in csv_files:
#     input_file_path = file_info['input_file']
#     output_file_path = file_info['output_file']

#     print(f"Processing file: {input_file_path}")

#     # Open the CSV file and prepare to write the cleaned output
#     with open(input_file_path, mode='r', newline='', encoding='utf-8') as infile, \
#          open(output_file_path, mode='w', newline='', encoding='utf-8') as outfile:
        
#         reader = csv.DictReader(infile)
#         fieldnames = reader.fieldnames  # Retain all original column headers
#         writer = csv.DictWriter(outfile, fieldnames=fieldnames)
#         writer.writeheader()  # Write the header to the output file

#         # Iterated through each row in the CSV
#         for row in reader:
#             # Clean  columns 
#             if 'question' in row:
#                 row['question'] = clean_text(row['question'])
#             if 'answer' in row:
#                 row['answer'] = clean_text(row['answer'])

#             # Write the cleaned row to the output file
#             writer.writerow(row)

#     print(f"Cleaned data saved to {output_file_path}")


# # load regularized csv
# df=pd.DataFrame(pd.read_csv('cleaned_Nlp_project_data.csv'))
# # print(df)
# df.dropna(inplace=True)

# df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# text = df["question"]
# tokenized_text = text.apply(lambda x: word_tokenize(str(x)))

# # print(tokenized_text)

# import spacy 
# nlp = spacy.load('en_core_web_sm')

# lemmatized_tokens = tokenized_text.apply(lambda x: [token.lemma_ for token in nlp(" ".join(x))])

# df["question"]= lemmatized_tokens.apply(lambda x: " ".join(x))

# # print(lemmatized_tokens)

# df["pos_tags_question"] = df["question"].apply(lambda x: [(token.text, token.pos_) for token in nlp(x)])

# # print(df['pos_tags'])

# text = df["answer"]
# # tokenized_text = text.apply(lambda x: word_tokenize(str(x)))

# # # print(tokenized_text)

# # import spacy 
# nlp = spacy.load('en_core_web_sm')

# # lemmatized_tokens = tokenized_text.apply(lambda x: [token.lemma_ for token in nlp(" ".join(x))])

# # df["answer"]= lemmatized_tokens.apply(lambda x: " ".join(x))

# # # print(lemmatized_tokens)

# df["pos_tags_answer"] = df["answer"].apply(lambda x: [(token.text, token.pos_) for token in nlp(x)])

# # # print(df['pos_tags_answer'])
# # df.to_csv('pre_processed_Nlp_project_data.csv', index=False)



import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')  # For tokenization
# nltk.download('stopwords')  # For stopword list
# nltk.download('wordnet')  # For WordNet lemmatization

df=pd.DataFrame(pd.read_csv("pre_processed_Nlp_project_data.csv"))



import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources (run this once)
# nltk.download('punkt')

# Read the CSV file into a DataFrame

# Define the tokenization function
def tokenize_text(text):
    if isinstance(text, str):  # Check if the input is a string
        return word_tokenize(text.lower())  # Tokenize and convert to lowercase
    return text  # Return the original value if it's not a string

# Apply the tokenization function to all columns
df = df.map(tokenize_text)

# Display the DataFrame with tokenized text
# print(df.head())

    # Remove stopwords
    # filtered_tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize the words
    # lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    # return ' '.join(lemmatized_tokens)  # Return the cleaned text


# Load GloVe embeddings

import os

glove_path = 'glove.6B.100d.txt'

embedding_dict = {}

with open(glove_path, 'r', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        vector = list(map(float, values[1:]))
        embedding_dict[word] = vector

def get_embedding(text):
    # Check if the input is a list
    if isinstance(text, list):
        # If it's a list, join it into a single string
        text = ' '.join(text)
    
    words = text.split()
    vectors = [embedding_dict[word] for word in words if word in embedding_dict]
    
    if vectors:
        # Sum the vectors element-wise
        summed_vector = [sum(x) for x in zip(*vectors)]
        # Return the average vector
        return [x / len(vectors) for x in summed_vector]
    else:
        return [0] * 100  # Return a zero vector if no words are found
# def get_embedding(text):
#     # Check if the  is a list
#     if isinstance(text, list):
#         # If it's a list, join it into a single string
#         text = ' '.join(text)
    
#     words = text.split()
#     vectors = [embedding_dict[word] for word in words if word in embedding_dict]
    
#     # Return the average vector or a zero vector if no words are found
#     return sum(vectors) / len(vectors) if vectors else [0] * 100


# df = df.map(get_embedding)
from sklearn.model_selection import train_test_split
X=df.drop(['pos_tags_answer', 'pos_tags_question','answer'],axis=1)
# print(X)
y=df["answer"]

xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42)



# implement model
if __name__ == '__main__':
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
    from torch.utils.data import DataLoader, Dataset
    
    # Initialize the tokenizer for DialoGPT
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    
    # Example of your tokenized xtrain (questions) and ytrain (answers)
    # xtrain = ["How are you?", "What is your name?", "Tell me a joke!"]
    # ytrain = ["I'm good, thanks!", "I'm DialoGPT, a chatbot!", "Why don't skeletons fight each other? They don't have the guts!"]
    
    # Tokenize questions and answers using the tokenizer
    def tokenize_data(questions, answers):
        inputs, labels = [], []
        for q, a in zip(questions, answers):
            # Ensure that q and a are strings before concatenating
            q = str(q)
            a = str(a)
            
            # Tokenize the question and answer, adding the EOS token
            input_ids = tokenizer.encode(q + tokenizer.eos_token + a, truncation=True, max_length=128)
            inputs.append(input_ids)
            labels.append(input_ids)  # For DialoGPT, the input is the same as the output
        return inputs, labels
    
    # Tokenizing the questions and answers
    train_inputs, train_labels = tokenize_data(xtrain, ytrain)
    
    # Padding the sequences
    def pad_data(inputs, labels):
        # Check if the tokenizer has a padding token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0  # Set to 0 if None (or use a suitable value)
    
        # Pad the inputs and labels
        padded_inputs = pad_sequence([torch.tensor(seq) for seq in inputs], batch_first=True, padding_value=tokenizer.pad_token_id)
        padded_labels = pad_sequence([torch.tensor(seq) for seq in labels], batch_first=True, padding_value=tokenizer.pad_token_id)
        
        return padded_inputs, padded_labels
    
    
    # Pad the tokenized inputs and labels
    train_inputs, train_labels = pad_data(train_inputs, train_labels)
    
    # Create a custom Dataset class
    class ChatDataset(Dataset):
        def __init__(self, inputs, labels):
            self.inputs = inputs
            self.labels = labels
    
        def __len__(self):
            return len(self.inputs)
    
        def __getitem__(self, idx):
            return {'input_ids': self.inputs[idx], 'labels': self.labels[idx]}
    
    # Create Dataset and DataLoader
    train_dataset = ChatDataset(train_inputs, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    # Load the pre-trained DialoGPT model
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    
    # Set up optimizer and device (use CPU if no GPU available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Training loop
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            # Move batch to the correct device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
    
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save the model after training
    model.save_pretrained("./chatbot_model")
    tokenizer.save_pretrained("./chatbot_model")
    
