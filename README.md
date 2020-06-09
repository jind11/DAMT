# DAMT
A new method for semi-supervised domain adaptation of Neural Machine Translation (NMT)

This is the source code for the paper: [Jin, D., Jin, Z., Zhou, J.T., & Szolovits, P. (2020). Unsupervised Domain Adaptation for Neural Machine Translation with Iterative Back Translation. ArXiv, abs/2001.08140.](https://arxiv.org/abs/2001.08140). If you use the code, please cite the paper:

```
@article{Jin2020UnsupervisedDA,
  title={Unsupervised Domain Adaptation for Neural Machine Translation with Iterative Back Translation},
  author={Di Jin and Zhijing Jin and Joey Tianyi Zhou and Peter Szolovits},
  journal={ArXiv},
  year={2020},
  volume={abs/2001.08140}
}
```

## Prerequisites:
Run the following command to install the prerequisite packages:
```
pip install -r requirements.txt
```
You should also install Moses tokenizer and fastBPE tool in the folder of "tools" by running the following commands:
```
cd tools
git clone https://github.com/moses-smt/mosesdecoder
git clone https://github.com/glample/fastBPE
cd fastBPE
g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
cd ../..
```

## Data:
Please download the data from the [Google Drive](https://drive.google.com/file/d/1aQOXfcGpPbQemG4mQQuiy6ZrCRn6WiDj/view?usp=sharing) and unzip it to the main directory of this repository. The data downloaded include the domains of MED (EMEA), IT, LAW (ACQUIS), and TED for DE-EN language pair and MED, LAW, and TED for EN-RO language pair. WMT14 DE-EN data can be downloaded [here](https://nlp.stanford.edu/projects/nmt/) and WMT16 EN-RO data is downloaded from [here](https://www.statmt.org/wmt16/translation-task.html).

## How to use
1. First we need to download the pretrained model parameter files from the [XLM repository](https://github.com/facebookresearch/XLM#pretrained-xlmmlm-models).

2. Then we need to process the data. Suppose we want to train the NMT model for the IT domain from German (de) to English (en), then run the following command:
```
./get-data-nmt-local.sh --src de --tgt en --data_name it --data_path ./data/de-en/it --reload_codes PATH_TO_PRETRAINED_MODEL_CODES --reload_vocab PATH_TO_PRETRAINED_MODEL_VOCAB
```

3. After data processing, to reproduce the "IBT" setting as mentioned in the paper, run the following command:
```
./train_IBT.sh --src de --tgt en --data_name it --pretrained_model_dir DIR_TO_PRETRAINED_MODEL
```

4. To reproduce the "IBT+SRC" setting, suppose we want to adapt from the Law domain to IT domain, where the source domain is Law (dataset name is acquis) and the target domain is IT, then run the following command:
```
./train_IBT_plus_SRC.sh --src de --tgt en --src_data_name acquis --tgt_data_name it --pretrained_model_dir DIR_TO_PRETRAINED_MODEL
```

5. In order to reproduce the "IBT+Back" setting, we need to go through several steps. 

* First of all, we need to train a NMT model to translate from en to de using the source domain data (acquis) by running the following command:
```
./train_sup.sh --src en --tgt de --data_name acquis --pretrained_model_dir DIR_TO_PRETRAINED_MODEL
```

   * After training this model, we get the translation results by using thie model to translate the English sentences in the target domain (it) to German, which are used as the back-translated data:
```
./translate_exe.sh --src en --tgt de --data_name it --model_name acquis --model_dir DIR_TO_TRAINED_MODEL
./get-data-back-translate.sh --src en --tgt de --data_name it --model_name acquis
```

   * When the back-translated data is ready, we can finally run this command:
```
./train_IBT_plus_BACK.sh --src de --tgt en --src_data_name acquis --tgt_data_name it --pretrained_model_dir DIR_TO_PRETRAINED_MODEL
```
