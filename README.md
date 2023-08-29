# pyt_tranfomer_finetuning
a script to fine tune transfomer models to predict cvss v2 labels
This code represents a part of my master's degree project in Natural Language Processing.
The project focuses on multi-class text classification using state-of-the-art Transformer-based models.
The code leverages the PyTorch framework and the Hugging Face Transformers library.

It involves fine-tuning a pre-trained language model, such as 'roberta-base' -which can be exchanged with any other transfomer model-,
for classifying text descriptions into multiple categories. 
The training pipeline encompasses data preprocessing, dataset creation, model initialization, and optimization. 
To achieve optimal results, the code implements techniques like label encoding, class weighting, and mixed precision training.

Throughout the training process, the code monitors the model's performance on a validation set, employing early stopping to prevent overfitting.
Post-training, the model's effectiveness is evaluated on a test dataset using various metrics including accuracy, precision, recall, F1-score.
The results are saved in CSV files for further analysis and interpretation.

This code showcases my dedication to advancing my expertise in NLP and machine learning, 
exemplifying the proficiency gained during my master's journey. Its successful execution demonstrates my ability to handle complex NLP tasks 
and provides a foundation for future projects in the domain.
