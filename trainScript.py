#modelName = "bert-base-uncased"
modelName = "dmis-lab/biobert-base-cased-v1.1"
trainDataPath = 'trainData.json'
evalDataPath = 'evalData.json'
categoriesListPath = 'listaCategorias.txt'
outputModelName = "modelPaper2"
lr = 1e-5
batchSize = 56


import torch
import wandb
import numpy as np
from datasets import Dataset
from typing import Optional, Union, Tuple
from torch.nn import BCEWithLogitsLoss, Dropout, Linear
from sklearn.metrics import multilabel_confusion_matrix
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.utils import add_start_docstrings_to_model_forward
from transformers.models.bert.modeling_bert import BERT_INPUTS_DOCSTRING
from transformers import TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, BertPreTrainedModel, BertModel, AutoConfig, AutoTokenizer


wandb_api = 'your_api_key'
wandb.login(key=wandb_api)

dTrain = Dataset.from_json(trainDataPath)
dEval  = Dataset.from_json(evalDataPath)


listaCategorias = []

with open(categoriesListPath, 'r') as f:
    for line in f:
        listaCategorias.append(line.strip())


tag2id = {'O': 0}
i = 1


for element in listaCategorias:
    tag2id[element] = i
    i += 1


#Create a tag2id array, having all the items in listaCategorias
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


MAX_LENGTH = 512


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


tokenizer = AutoTokenizer.from_pretrained(modelName)
tokenized_data_train = dTrain.map(tokenize_and_adjust_labels, remove_columns=dTrain.column_names)
tokenized_data_eval  = dEval.map(tokenize_and_adjust_labels,  remove_columns=dEval.column_names)


data_collator = DataCollatorWithPadding(tokenizer = tokenizer, padding = True)


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


n_labels = len(id2label)


def divide(a: int, b: int):
    return a / b if b > 0 else 0


def compute_metrics(p):

    predictions, true_labels = p
    predicted_labels = np.where(predictions > 0, np.ones(predictions.shape), np.zeros(predictions.shape))
    metrics = {}
    cm = multilabel_confusion_matrix(true_labels.reshape(-1, n_labels), predicted_labels.reshape(-1, n_labels))
    
    for label_idx, matrix in enumerate(cm):
        if label_idx == 0:
            continue
        tp, fp, fn = matrix[1, 1], matrix[0, 1], matrix[1, 0]
        precision = divide(tp, tp + fp)
        recall = divide(tp, tp + fn)
        f1 = divide(2 * precision * recall, precision + recall)
        metrics[f"f1_{id2label[label_idx]}"] = f1
        
    macro_f1 = sum(list(metrics.values())) / (n_labels - 1)
    metrics["macro_f1"] = macro_f1
        
    return metrics


training_args = TrainingArguments(
    output_dir=outputModelName + '_args',
    evaluation_strategy="epoch",
    learning_rate=lr,
    per_device_train_batch_size=batchSize,
    per_device_eval_batch_size=batchSize,
    per_gpu_train_batch_size=batchSize,
    per_gpu_eval_batch_size=batchSize,
    num_train_epochs=30,
    weight_decay=0.2,
    logging_steps = 1000,
    save_strategy='epoch',
    save_total_limit=30,
    load_best_model_at_end=True,
    metric_for_best_model='macro_f1',
    log_level='critical',
    seed=12345
)


def model_init():
    config = AutoConfig.from_pretrained(modelName, num_labels=len(id2label), id2label=id2label, label2id=label2id)
    return BertForSpanClassification.from_pretrained(modelName, config=config)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_data_train,
    eval_dataset=tokenized_data_eval,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()


trainer.model.save_pretrained(outputModelName)