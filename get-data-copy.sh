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
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"

if [ "$SRC" \< "$TGT" ]; then
    ORDERED_SRC=$SRC
    ORDERED_TGT=$TGT
else
    ORDERED_SRC=$TGT
    ORDERED_TGT=$SRC
fi

MAIN_PATH=$PWD
DATA_PATH=/data/medg/misc/jindi/nlp/datasets/OPUS/$ORDERED_SRC-$ORDERED_TGT/$SRC_DATA_NAME
PROC_PATH=$DATA_PATH/processed/$ORDERED_SRC-$ORDERED_TGT
COPY_DATA_DIR=$DATA_PATH/copy/$SRC_MODEL_NAME
FULL_VOCAB=$PROC_PATH/vocab.$ORDERED_SRC-$ORDERED_TGT
SRC_MODEL_PATH=/data/medg/misc/jindi/nlp/datasets/OPUS/$ORDERED_SRC-$ORDERED_TGT/$SRC_MODEL_NAME/processed/$ORDERED_SRC-$ORDERED_TGT

mkdir -p $COPY_DATA_DIR

cat $SRC_MODEL_PATH/train.$ORDERED_SRC-$ORDERED_TGT.$SRC $PROC_PATH/train.$ORDERED_SRC-$ORDERED_TGT.$TGT > $COPY_DATA_DIR/train.$ORDERED_SRC-$ORDERED_TGT.$SRC
cat $SRC_MODEL_PATH/train.$ORDERED_SRC-$ORDERED_TGT.$TGT $PROC_PATH/train.$ORDERED_SRC-$ORDERED_TGT.$TGT > $COPY_DATA_DIR/train.$ORDERED_SRC-$ORDERED_TGT.$TGT
$MAIN_PATH/preprocess.py $FULL_VOCAB $COPY_DATA_DIR/train.$ORDERED_SRC-$ORDERED_TGT.$SRC
$MAIN_PATH/preprocess.py $FULL_VOCAB $COPY_DATA_DIR/train.$ORDERED_SRC-$ORDERED_TGT.$TGT