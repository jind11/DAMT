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
  --data_name)
    DATA_NAME="$2"; shift 2;;
  --model_name)
    MODEL_NAME="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"

if [ "$SRC" \> "$TGT" ]; then echo "please ensure SRC < TGT"; exit; fi

MAIN_PATH=$PWD
DATA_PATH=data/$SRC-$TGT/$DATA_NAME
PROC_PATH=$DATA_PATH/processed/$SRC-$TGT
BACK_DATA_DIR=$DATA_PATH/back_translate/$MODEL_NAME
FULL_VOCAB=$PROC_PATH/vocab.$SRC-$TGT

$MAIN_PATH/preprocess.py $FULL_VOCAB $BACK_DATA_DIR/train.$SRC-$TGT.$SRC
$MAIN_PATH/preprocess.py $FULL_VOCAB $BACK_DATA_DIR/train.$SRC-$TGT.$TGT
