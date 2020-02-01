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

if [ "$SRC" \< "$TGT" ]; then
    ORDERED_SRC=$SRC
    ORDERED_TGT=$TGT
else
    ORDERED_SRC=$TGT
    ORDERED_TGT=$SRC
fi

SRC_DIR=/data/medg/misc/jindi/nlp/model_results/DAMT/${TRAIN_TYPE}_$SRC_MODEL_NAME\_$SRC\_$TGT
OUT_DIR=/data/medg/misc/jindi/nlp/datasets/OPUS/$ORDERED_SRC-$ORDERED_TGT/$SRC_DATA_NAME/${TRAIN_TYPE}_back_translate/$SRC_MODEL_NAME
mkdir -p $OUT_DIR

python -W ignore translate.py \
    --exp_name $SRC_MODEL_NAME\_$SRC\_to_$TGT \
    --dump_path ./back_translate/ \
    --model_path $SRC_DIR/best-valid_$SRC-$TGT\_mt_bleu.pth \
    --src_data_path /data/medg/misc/jindi/nlp/datasets/OPUS/$ORDERED_SRC-$ORDERED_TGT/$SRC_DATA_NAME/processed/$ORDERED_SRC-$ORDERED_TGT/train.$SRC \
    --output_path_source $OUT_DIR/train.$ORDERED_SRC-$ORDERED_TGT.$SRC \
    --output_path_target $OUT_DIR/train.$ORDERED_SRC-$ORDERED_TGT.$TGT \
    --src_lang $SRC \
    --tgt_lang $TGT \
    --batch_size 128 \