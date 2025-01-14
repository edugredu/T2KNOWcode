testDataPath = 'testData.json'
categoriesPath = 'listaCategorias.txt'

import sys

if len(sys.argv) != 2:
    print('Usage: python evaluation.py modelPath')
    sys.exit(1)
    
modelPath = sys.argv[1]
print('Model path: ' + modelPath)
outputResults = modelPath + '/results'

import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from typing import Optional, Union, Tuple
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.utils import add_start_docstrings_to_model_forward
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer
from transformers.models.bert.modeling_bert import BERT_INPUTS_DOCSTRING
from torch.nn import BCEWithLogitsLoss, Dropout, Linear


dTest  = Dataset.from_json(testDataPath)


listaCategorias = []


with open(categoriesPath, 'r') as f:
    for line in f:
        listaCategorias.append(line.strip())


tag2id = {'O': 0}
i = 1

for element in listaCategorias:
    tag2id[element] = i
    i += 1

id2label = {0: 'O'}
i = 1

for element in listaCategorias:
    id2label[i] = 'B-' + element
    id2label[i+1] = 'I-' + element
    i += 2

label2id = {v: k for k, v in id2label.items()}


def get_token_role_in_span(token_start: int, token_end: int, span_start: int, span_end: int):

    if token_end <= token_start:
        return "N"
    if token_start < span_start or token_end > span_end:
        return "O"
    if token_start > span_start:
        return "I"
    else:
        return "B"


MAX_LENGTH = 256

def tokenize_and_adjust_labels(sample):

    # Tokenize the text, keep the start and end positions of tokens with `return_offsets_mapping` option
    # Use max_length and truncation to ajust the text length
    tokenized = tokenizer(sample["text"], 
                          return_offsets_mapping=True, 
                          padding="max_length", 
                          max_length=MAX_LENGTH,
                          truncation=True)
    
    # We are doing a multilabel classification task at each token, we create a list of size len(label2id)
    labels = [[0 for _ in label2id.keys()] for _ in range(MAX_LENGTH)]
    
    # Scan all the tokens and spans, assign 1 to the corresponding label if the token lies at the beginning
    # or inside the spans
    for (token_start, token_end), token_labels in zip(tokenized["offset_mapping"], labels):
        for span in sample["tags"]:
            role = get_token_role_in_span(token_start, token_end, span["start"], span["end"])
            if role == "B":
                token_labels[label2id[f"B-{span['tag']}"]] = 1
            elif role == "I":
                token_labels[label2id[f"I-{span['tag']}"]] = 1
    
    return {**tokenized, "labels": labels}


class BertForSpanClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = Dropout(classifier_dropout)
        self.classifier = Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


model = BertForSpanClassification.from_pretrained(modelPath).to('cuda')


with open(modelPath + "/config.json") as json_file:
    data = json.load(json_file)
    baseModel = data['_name_or_path']

tokenizer = AutoTokenizer.from_pretrained(baseModel)
tokenized_data_test = dTest.map(tokenize_and_adjust_labels, remove_columns=dTest.column_names)


def get_offsets_and_predicted_tags(text, model, tokenizer, threshold=0):

  # Tokenize the sentence to retrieve the tokens and offset mappings
  raw_encoded_example = tokenizer(text, return_offsets_mapping=True)
  encoded_example = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to('cuda')
  
  # Call the model. The output LxK-tensor where L is the number of tokens, K is the number of classes
  out = model(**encoded_example)["logits"][0]

  #confidences = torch.sigmoid(out).detach().numpy()
  confidences = torch.nn.functional.softmax(out, dim=-1)

  # Assign to each token the classes whose logit is positive
  predicted_tags = [[i for i, l in enumerate(logit) if l > threshold] for logit in out]
    
  results = []
  for token, tags, offset, confidence in zip(
        tokenizer.batch_decode(raw_encoded_example["input_ids"]),
        predicted_tags,
        raw_encoded_example["offset_mapping"],
        confidences
  ):
        result = {
            "token": token,
            "tags": tags,
            "offset": offset,
            "confidence": confidence.tolist()
        }
        results.append(result)
  return results


def get_tagged_groups(text, model, tokenizer, includeText = False, confidence=0.99):

    offsets_and_tags = get_offsets_and_predicted_tags(text, model, tokenizer)
    predicted_offsets = {l: [] for l in tag2id}
    last_token_tags = []
    deleted_tags = [] 

    for item in offsets_and_tags:
        (start, end), tags, conf, token = item["offset"], item["tags"], item["confidence"], item["token"]

        #Check if there are tags with I- that are contained in deleted_tags
        for tag in deleted_tags.copy():
            if label2id['I-'+tag] not in tags:
                deleted_tags.remove(tag)

        for label_id in tags.copy():
            label = id2label[label_id]
            tag = label[2:] # "I-PER" => "PER"
            if label.startswith("B-"):
                if conf[label_id] > confidence:
                    predicted_offsets[tag].append({"start": start, "end": end})
                else:
                    deleted_tags.append(tag)
            elif label.startswith("I-"):
                # If "B-" and "I-" both appear in the same tag, ignore as we already processed it
                if label2id[f"B-{tag}"] in tags:
                    continue
                
                # If the previous token was not tagged due to low confidence, ignore this token as well
                if tag in deleted_tags:
                    continue
                
                if label_id not in last_token_tags and label2id[f"B-{tag}"] not in last_token_tags:
                        predicted_offsets[tag].append({"start": start, "end": end})
                else:
                    predicted_offsets[tag][-1]["end"] = end
    
        last_token_tags = tags   

    if includeText:    
        flatten_predicted_offsets = [{**v, "tag": k, "text": text[v["start"]:v["end"]]} 
                                    for k, v_list in predicted_offsets.items() for v in v_list if v["end"] - v["start"] >= 3]
    else:
        flatten_predicted_offsets = [{**v, "tag": k} 
                                    for k, v_list in predicted_offsets.items() for v in v_list if v["end"] - v["start"] >= 3]
    flatten_predicted_offsets = sorted(flatten_predicted_offsets, 
                                       key = lambda row: (row["start"], row["end"], row["tag"]))

    return flatten_predicted_offsets


n_labels = len(id2label)


def printMetricsDf(df, title):

    message = ''

    message += title + '\n'
    message += 'Precision: ' + str(round(df[0]['Precision'],2)) + '\n'
    message += 'Recall: ' + str(round(df[0]['Recall'],2)) + '\n'
    message += 'F1: ' + str(round(df[0]['F1'],2)) + '\n'
    message += 'Accuracy: ' + str(round(df[0]['Accuracy'],2)) + '\n'

    return message


def evalDataset3(dataset, model, tokenizer, tag2id, confidence=0.99):

    df = pd.DataFrame(columns=['Ca', 'Ia', 'Pa', 'Ma', 'Sa', 'Precision', 'Recall', 'F1', 'Accuracy'], index=tag2id.keys())
    df = df.fillna(0)
    df = df.drop('O')

    for element in tqdm(dataset):
        text = element['text']
        tags_real = element['tags']
        tags_predicted = get_tagged_groups(text, model, tokenizer, confidence=confidence)
        textTag = {'real': tags_real, 'predicted': tags_predicted}
        df = detectAndClassifyTexts3(textTag, df)
        del tags_real, tags_predicted, textTag

    print("Calculating metrics...")
    print("Correct: " + str(df['Ca'].sum()))
    print("Partial: " + str(df['Pa'].sum()))
    print("Incorrect: " + str(df['Ia'].sum()))
    print("Missing: " + str(df['Ma'].sum()))
    print("Spurious: " + str(df['Sa'].sum()))


    #Calculate precision, recall, F1 and accuracy
    df['Precision'] = (df['Ca'] + 0.5 * df['Pa']) / (df['Ca'] + df['Ia'] + df['Pa'] + df['Sa'])
    df['Recall'] = (df['Ca'] + 0.5 * df['Pa']) / (df['Ca'] + df['Ia'] + df['Pa'] + df['Ma'])
    df['F1'] = 2 * (df['Precision'] * df['Recall']) / (df['Precision'] + df['Recall'])
    df['Accuracy'] = (df['Ca'] + 0.5 * df['Pa']) / (df['Ca'] + df['Ia'] + df['Pa'] + df['Ma'] + df['Sa'])

    #Delete rows where Ca,Ia, Pa, Ma and Sa are all 0
    df = df[(df.T != 0).any()]

    #Fill the NaN values with 1
    df = df.fillna(1)

    for row in df.index:
        if df.loc[row, 'Ca'] == 0 and df.loc[row, 'Pa'] == 0 and df.loc[row, 'Ia'] == 0 and df.loc[row, 'Ma'] == 0 and df.loc[row, 'Sa'] == 0:
            df = df.drop(row)

    precision = (df['Ca'].sum() + 0.5 * df['Pa'].sum()) / (df['Ca'].sum() + df['Ia'].sum() + df['Pa'].sum() + df['Sa'].sum())
    recall = (df['Ca'].sum() + 0.5 * df['Pa'].sum()) / (df['Ca'].sum() + df['Ia'].sum() + df['Pa'].sum() + df['Ma'].sum())
    F1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (df['Ca'].sum() + 0.5 * df['Pa'].sum()) / (df['Ca'].sum() + df['Ia'].sum() + df['Pa'].sum() + df['Ma'].sum() + df['Sa'].sum())

    #Get the average metrics of each dataframe
    infoMetrics = [{'Precision': precision, 'Recall': recall, 'F1': F1, 'Accuracy': accuracy}]
    
    return [df, infoMetrics]

def detectAndClassifyTexts3(textTag, df):

    #Order the real and predicted arrays by start, and if start is the same, by tag
    real      = sorted(textTag['real'],      key = lambda row: (row["start"], row["tag"]))
    predicted = sorted(textTag['predicted'], key = lambda row: (row["start"], row["tag"]))

    #CORRECT - Ca
    for annR in real.copy():
        for annP in predicted.copy():
            if annR['start'] == annP['start'] and annR['end'] == annP['end'] and annR['tag'] == annP['tag']:
                
                df.loc[annR['tag'], 'Ca'] += 1

                real.remove(annR)
                predicted.remove(annP)

                break

    #PARTIAL - Ca
    for annR in real.copy():
        #Get all the annotation within the range of the real annotation
        partialMatch = []
        
        for annP in predicted.copy():
            if (annP['start'] >= annR['start'] and annP['end'] <= annR['end']) or \
            (annP['start'] <= annR['start'] and annP['end'] >= annR['start'] and annP['end'] <= annR['end']) or \
            (annP['start'] >= annR['start'] and annP['start'] <= annR['end'] and annP['end'] >= annR['end']) or \
            (annP['start'] <= annR['start'] and annP['end'] >= annR['end']):
                
                #Append the element to the partialMatch array
                partialMatch.append(annP)

        if len(partialMatch) != 0:
                
                #Order the partialMatch elements by size (end - start)
                partialMatch = sorted(partialMatch, key = lambda row: (row["end"] - row["start"]))

                #Check if some of the element tags are the same as the real annotation
                for annP in partialMatch:
                    if annP['tag'] == annR['tag']:
                        df.loc[annR['tag'], 'Pa'] += 1
                        real.remove(annR)
                        predicted.remove(annP)
                        break
                    
    #INCORRECT - Ia
    for annR in real.copy():
        for annP in predicted.copy():
            if annP['start'] == annR['start'] and annP['end'] == annR['end'] and annP['tag'] != annR['tag']:
                
                df.loc[annR['tag'], 'Ia'] += 1

                real.remove(annR)
                predicted.remove(annP)

                break

    #MISSING - Ma
    for annR in real.copy():
        df.loc[annR['tag'], 'Ma'] += 1


    #SPURIOUS - Sa
    for annP in predicted:
        df.loc[annP['tag'], 'Sa'] += 1

    del real, predicted
    
    return df


dfTest, metricsTest  = evalDataset3(dTest,  model, tokenizer, tag2id)


print(printMetricsDf(metricsTest, "Metrics test"))


if not os.path.exists(outputResults):
    os.makedirs(outputResults)


with open(outputResults + '/metricsTest.txt', 'w') as f:
    f.write(printMetricsDf(metricsTest, "Metrics test"))


dfTest.to_csv(outputResults + '/dfTestResults.csv', index=True)