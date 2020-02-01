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
  --tgt_data_name)
    TGT_DATA_NAME="$2"; shift 2;;
  --sample_size)
    SAMPLE_SIZE="$2"; shift 2;;
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
DATA_PATH=/data/medg/misc/jindi/nlp/datasets/OPUS/$ORDERED_SRC-$ORDERED_TGT/$TGT_DATA_NAME
PROC_PATH=$DATA_PATH/processed/$ORDERED_SRC-$ORDERED_TGT
OUT_DATA_DIR=$DATA_PATH/processed_subsample_${SAMPLE_SIZE}_merge_${SRC_DATA_NAME}
FULL_VOCAB=$PROC_PATH/vocab.$ORDERED_SRC-$ORDERED_TGT

mkdir -p $OUT_DATA_DIR
python sub_sample_and_merge.py $TGT_DATA_NAME $SRC_DATA_NAME $SRC $TGT $SAMPLE_SIZE $OUT_DATA_DIR

$MAIN_PATH/preprocess.py $FULL_VOCAB $OUT_DATA_DIR/train.$ORDERED_SRC-$ORDERED_TGT.$SRC
$MAIN_PATH/preprocess.py $FULL_VOCAB $OUT_DATA_DIR/train.$ORDERED_SRC-$ORDERED_TGT.$TGT

ln -sf $PROC_PATH/valid.$ORDERED_SRC-$ORDERED_TGT.$SRC.pth $OUT_DATA_DIR/valid.$ORDERED_SRC-$ORDERED_TGT.$SRC.pth
ln -sf $PROC_PATH/valid.$ORDERED_SRC-$ORDERED_TGT.$TGT.pth $OUT_DATA_DIR/valid.$ORDERED_SRC-$ORDERED_TGT.$TGT.pth
ln -sf $PROC_PATH/test.$ORDERED_SRC-$ORDERED_TGT.$SRC.pth $OUT_DATA_DIR/test.$ORDERED_SRC-$ORDERED_TGT.$SRC.pth
ln -sf $PROC_PATH/test.$ORDERED_SRC-$ORDERED_TGT.$TGT.pth $OUT_DATA_DIR/test.$ORDERED_SRC-$ORDERED_TGT.$TGT.pth