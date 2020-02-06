# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

SRC=en
TGT=si

BPESIZE=5000
TRAIN_MINLEN=6  # remove sentences with <6 BPE tokens
TRAIN_MAXLEN=150  # remove sentences with >250 BPE tokens

ROOT=$(dirname "$0")
SCRIPTS=$ROOT/scripts
DATA=/home/ubuntu/proj/data/flores
#TMP=$DATA/wiki_${SRC}_${TGT}_bpe${BPESIZE}
OUT_PATH=/home/ubuntu/proj/data/$SRC-$TGT
DATABIN=$OUT_PATH/data_bin
PROC_PATH=$OUT_PATH/processed/$SRC-$TGT
mkdir -p $PROC_PATH

# BPE / vocab files
BPE_CODES=$PROC_PATH/codes
SRC_VOCAB=$PROC_PATH/vocab.$SRC
TGT_VOCAB=$PROC_PATH/vocab.$TGT
FULL_VOCAB=$PROC_PATH/vocab.$SRC-$TGT

TGT_TOKENIZER="bash $SCRIPTS/indic_norm_tok.sh $SRC"
SRC_TOKENIZER="cat"  # learn target-side BPE over untokenized (raw) text
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py

MAIN_PATH=/home/ubuntu/proj/DAMT
TOOLS_PATH=$MAIN_PATH/tools
FASTBPE=$TOOLS_PATH/fastBPE/fast

# raw data
PARA_SRC_TRAIN=$OUT_PATH/train.$SRC-$TGT.$SRC
PARA_TGT_TRAIN=$OUT_PATH/train.$SRC-$TGT.$TGT
PARA_SRC_VALID=$OUT_PATH/valid.$SRC-$TGT.$SRC
PARA_TGT_VALID=$OUT_PATH/valid.$SRC-$TGT.$TGT
PARA_SRC_TEST=$OUT_PATH/test.$SRC-$TGT.$SRC
PARA_TGT_TEST=$OUT_PATH/test.$SRC-$TGT.$TGT

# BPE data
PARA_SRC_TRAIN_BPE=$PROC_PATH/train.$SRC-$TGT.$SRC
PARA_TGT_TRAIN_BPE=$PROC_PATH/train.$SRC-$TGT.$TGT
PARA_SRC_VALID_BPE=$PROC_PATH/valid.$SRC-$TGT.$SRC
PARA_TGT_VALID_BPE=$PROC_PATH/valid.$SRC-$TGT.$TGT
PARA_SRC_TEST_BPE=$PROC_PATH/test.$SRC-$TGT.$SRC
PARA_TGT_TEST_BPE=$PROC_PATH/test.$SRC-$TGT.$TGT

URLS=(
    "https://github.com/facebookresearch/flores/raw/master/data/wikipedia_en_ne_si_test_sets.tgz"
)
ARCHIVES=(
    "wikipedia_en_ne_si_test_sets.tgz"
)
TRAIN_SETS=(
    "all-clean-si/GNOMEKDEUbuntu.en-si"
    "all-clean-si/OpenSubtitles2018.en-si"
)
VALID_SET="wikipedia_en_ne_si_test_sets/wikipedia.dev.si-en"
TEST_SET="wikipedia_en_ne_si_test_sets/wikipedia.devtest.si-en"

if [ ! -d $DATA/all-clean-si ]; then
    echo "Data directory not found. Please run 'bash download-data.sh' first..."
    exit -1
fi

# download and extract data
for ((i=0;i<${#URLS[@]};++i)); do
    ARCHIVE=$DATA/${ARCHIVES[i]}
    if [ -f $ARCHIVE ]; then
        echo "$ARCHIVE already exists, skipping download"
    else
        URL=${URLS[i]}
        wget -P $DATA "$URL"
        if [ -f $ARCHIVE ]; then
            echo "$URL successfully downloaded."
        else
            echo "$URL not successfully downloaded."
            exit -1
        fi
    fi
    FILE=${ARCHIVE: -4}
    if [ -e $FILE ]; then
        echo "$FILE already exists, skipping extraction"
    else
        tar -C $DATA -xzvf $ARCHIVE
    fi
done

echo "pre-processing train data..."
bash $SCRIPTS/download_indic.sh
for FILE in "${TRAIN_SETS[@]}" ; do
    $SRC_TOKENIZER $DATA/$FILE.$SRC
done > $PARA_SRC_TRAIN
for FILE in "${TRAIN_SETS[@]}"; do
    $TGT_TOKENIZER $DATA/$FILE.$TGT
done > $PARA_TGT_TRAIN

echo "pre-processing dev/test data..."
$SRC_TOKENIZER $DATA/${VALID_SET}.$SRC > $PARA_SRC_VALID
$TGT_TOKENIZER $DATA/${VALID_SET}.$TGT > $PARA_TGT_VALID
$SRC_TOKENIZER $DATA/${TEST_SET}.$SRC > $PARA_SRC_TEST
$TGT_TOKENIZER $DATA/${TEST_SET}.$TGT > $PARA_TGT_TEST


# learn BPE with sentencepiece
python $SPM_TRAIN \
  --input=$PARA_SRC_TRAIN,$PARA_TGT_TRAIN \
  --model_prefix=$PROC_PATH/sentencepiece.bpe \
  --vocab_size=$BPESIZE \
  --character_coverage=1.0 \
  --model_type=bpe

# encode train/valid/test
python $SPM_ENCODE \
  --model $PROC_PATH/sentencepiece.bpe.model \
  --output_format=piece \
  --inputs $PARA_SRC_TRAIN $PARA_TGT_TRAIN \
  --outputs $PARA_SRC_TRAIN_BPE $PARA_TGT_TRAIN_BPE \
  --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN
python $SPM_ENCODE \
  --model $PROC_PATH/sentencepiece.bpe.model \
  --output_format=piece \
  --inputs $PARA_SRC_VALID $PARA_TGT_VALID \
  --outputs $PARA_SRC_VALID_BPE $PARA_TGT_VALID_BPE
python $SPM_ENCODE \
  --model $PROC_PATH/sentencepiece.bpe.model \
  --output_format=piece \
  --inputs $PARA_SRC_TEST $PARA_TGT_TEST \
  --outputs $PARA_SRC_TEST_BPE $PARA_TGT_TEST_BPE

# extract full vocabulary
if ! [[ -f "$FULL_VOCAB" ]]; then
  echo "Extracting vocabulary..."
  $FASTBPE getvocab $PARA_SRC_TRAIN_BPE $PARA_TGT_TRAIN_BPE > $FULL_VOCAB
fi
echo "Full vocab in: $FULL_VOCAB"

# binarize data
if ! [[ -f "$PARA_SRC_TRAIN_BPE.pth" ]]; then
  echo "Binarizing $SRC data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_SRC_TRAIN_BPE
fi
if ! [[ -f "$PARA_TGT_TRAIN_BPE.pth" ]]; then
  echo "Binarizing $TGT data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_TGT_TRAIN_BPE
fi
echo "$SRC binarized data in: $PARA_SRC_TRAIN_BPE.pth"
echo "$TGT binarized data in: $PARA_TGT_TRAIN_BPE.pth"

if ! [[ -f "$PARA_SRC_VALID_BPE.pth" ]]; then
  echo "Binarizing $SRC data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_SRC_VALID_BPE
fi
if ! [[ -f "$PARA_TGT_VALID_BPE.pth" ]]; then
  echo "Binarizing $TGT data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_TGT_VALID_BPE
fi
echo "$SRC binarized data in: $PARA_SRC_VALID_BPE.pth"
echo "$TGT binarized data in: $PARA_TGT_VALID_BPE.pth"

if ! [[ -f "$PARA_SRC_TEST_BPE.pth" ]]; then
  echo "Binarizing $SRC data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_SRC_TEST_BPE
fi
if ! [[ -f "$PARA_TGT_TEST_BPE.pth" ]]; then
  echo "Binarizing $TGT data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_TGT_TEST_BPE
fi
echo "$SRC binarized data in: $PARA_SRC_TEST_BPE.pth"
echo "$TGT binarized data in: $PARA_TGT_TEST_BPE.pth"

# binarize data
fairseq-preprocess \
  --source-lang $SRC --target-lang $TGT \
  --trainpref $PROC_PATH/train.$SRC-$TGT \
  --validpref $PROC_PATH/valid.$SRC-$TGT \
  --testpref $PROC_PATH/test.$SRC-$TGT \
  --destdir $DATABIN \
  --joined-dictionary \
  --workers 4

