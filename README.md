## Exp1: Examine how UMT performs with parallel and unparalleled corpus 
I randomly created four sets of parallel and unparalled monolingual datasets for TED (de-en) and want to compared the performance of UMT between the two sets, so as to see whether the existence of parallel sentences can affect the UMT or not.

### Steps to run this exp
1. In my code, I used pre-trained parameters for initialization, so we need to first download these parameters from [here](https://github.com/facebookresearch/XLM#pretrained-cross-lingual-language-models). Currently, we only need the en-de pair. 

2. I have prepared the data for training, which is stored in my server, and the folder name is: /data/medg/misc/jindi/nlp/datasets/OPUS/de-en/TED_para_vs_unpara. There are 8 folders within it and it is easy to know what they are used for from the folder name. 

3. Execute the following commands:

```
./train_de_en_OPUS.sh --src de --tgt en --data_name TED_para_vs_unpara/seed_1_para
./train_de_en_OPUS.sh --src de --tgt en --data_name TED_para_vs_unpara/seed_1_unpara
./train_de_en_OPUS.sh --src de --tgt en --data_name TED_para_vs_unpara/seed_2_para
./train_de_en_OPUS.sh --src de --tgt en --data_name TED_para_vs_unpara/seed_2_unpara
./train_de_en_OPUS.sh --src de --tgt en --data_name TED_para_vs_unpara/seed_3_para
./train_de_en_OPUS.sh --src de --tgt en --data_name TED_para_vs_unpara/seed_3_unpara
./train_de_en_OPUS.sh --src de --tgt en --data_name TED_para_vs_unpara/seed_4_para
./train_de_en_OPUS.sh --src de --tgt en --data_name TED_para_vs_unpara/seed_4_unpara
```

In the script of ''train_de_en_OPUS.sh'', the variable ''pretrained_model_dir'' denotes the directory of the pre-trained model parameters, and the variable ''data_dir'' is where you put the data. All other parameters should be familiar by you already.

4. After all experments have ended, the results are stored in the folder of ''umt_$DATA_NAME\_$SRC\_$TGT'' and we can use this script to get the results:

```
python results_retrieval.py [folder_name] de en
```

Here ''folder_name'' means the folder name of the stored results. 

Let me know if you have any questions.

## Exp2: This experiment examins the self-training strategy
Steps to run:

1. Get data from the folder for the TED dataset: /data/medg/misc/jindi/nlp/datasets/OPUS/de-en/TED, and from this folder for the EMEA dataset: /data/medg/misc/jindi/nlp/datasets/OPUS/de-en/EMEA.

2. Run the following commands:

```
./train_de_en_OPUS_sup_back.sh --src de --tgt en --src_data_name WMT14 --tgt_data_name TED --train_type forth
./train_de_en_OPUS_sup_back.sh --src de --tgt en --src_data_name WMT14 --tgt_data_name TED --train_type forth_back
./train_de_en_OPUS_sup_back.sh --src de --tgt en --src_data_name WMT14 --tgt_data_name EMEA --train_type forth
./train_de_en_OPUS_sup_back.sh --src de --tgt en --src_data_name WMT14 --tgt_data_name EMEA --train_type forth_back
```

3. Retrieve results using the script abovementioned.
