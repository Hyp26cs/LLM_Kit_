
### KGRL 

Dataset and source code for the anonymous submission KGRL.


### Requirements

— python==3.8

— torch-geometric==2.0.1

— other pakages

To install requirements, please run `pip install -r requirements.txt.`


### Environment

— OS: CentOS Linux release 7.7.1908

— CPU: 64 Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz

— GPU: Four Tesla V100-SXM2 32GB

— CUDA: 10.1


### Running

**Prepare the Pre-trained model :**

— 1. Get the RoBERTa pre-trained model from [Website](https://huggingface.co/roberta-base);



— 2. Put the pre-trained model to the coresponseding path (pretrained);



**Process the dataset :** 

 —  python process_data.py 

 —— we have preprocessed the data, you can skip this step！


**Train the model :** 

 —  python train.py or CUDA_VISIBLE_DEVICES=0,1 python train.py










