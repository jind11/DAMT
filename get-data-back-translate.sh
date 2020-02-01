# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

#
# Read arguments
#
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --src)
    SRC="$2"; shift 2;;
  --tgt)
    TGT="$2"; shift 2;;
  --src_data_name)
    SRC_DATA_NAME="$2"; shift 2;;
  --src_model_name)
    SRC_MODEL_NAME="$2"; shift 2;;
  --train_type)
    TRAIN_TYPE="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"

if [ "$SRC" \> "$TGT" ]; then echo "please ensure SRC < TGT"; exit; fi

MAIN_PATH=$PWD
DATA_PATH=/data/medg/misc/jindi/nlp/datasets/OPUS/$SRC-$TGT/$SRC_DATA_NAME
PROC_PATH=$DATA_PATH/processed/$SRC-$TGT
BACK_DATA_DIR=$DATA_PATH/${TRAIN_TYPE}_back_translate/$SRC_MODEL_NAME
BACK_PLUS_DATA_DIR=$DATA_PATH/${TRAIN_TYPE}_back_translate/$SRC_MODEL_NAME\_plus
FULL_VOCAB=$PROC_PATH/vocab.$SRC-$TGT

$MAIN_PATH/preprocess.py $FULL_VOCAB $BACK_DATA_DIR/train.$SRC-$TGT.$SRC
$MAIN_PATH/preprocess.py $FULL_VOCAB $BACK_DATA_DIR/train.$SRC-$TGT.$TGT

#mkdir -p $BACK_PLUS_DATA_DIR

#cat /data/medg/misc/jindi/nlp/datasets/OPUS/$SRC-$TGT/$SRC_MODEL_NAME/processed/$SRC-$TGT/train.$SRC-$TGT.$SRC $BACK_DATA_DIR/train.$SRC-$TGT.$SRC > $BACK_PLUS_DATA_DIR/train.$SRC-$TGT.$SRC
#cat /data/medg/misc/jindi/nlp/datasets/OPUS/$SRC-$TGT/$SRC_MODEL_NAME/processed/$SRC-$TGT/train.$SRC-$TGT.$TGT $BACK_DATA_DIR/train.$SRC-$TGT.$TGT > $BACK_PLUS_DATA_DIR/train.$SRC-$TGT.$TGT
#$MAIN_PATH/preprocess.py $FULL_VOCAB $BACK_PLUS_DATA_DIR/train.$SRC-$TGT.$SRC
#$MAIN_PATH/preprocess.py $FULL_VOCAB $BACK_PLUS_DATA_DIR/train.$SRC-$TGT.$TGT