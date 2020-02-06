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
  --result_name)
    RESULT_NAME="$2"; shift 2;;
  --bleu_type)
    BLEU_TYPE="$2"; shift 2;;
  --set_type)
    SET_TYPE="$2"; shift 2;;
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

if [ $BLEU_TYPE == "sacrebleu" ]; then
python generate.py \
    /home/ubuntu/proj/data/$ORDERED_SRC-$ORDERED_TGT/data_bin/ \
    --source-lang $SRC --target-lang $TGT \
    --path ${RESULT_NAME}/checkpoint_best.pt \
    --beam 5 --lenpen 1.2 \
    --gen-subset $SET_TYPE \
    --remove-bpe=sentencepiece \
    --sacrebleu
else
python generate.py \
    /home/ubuntu/proj/data/$ORDERED_SRC-$ORDERED_TGT/data_bin/ \
    --source-lang $SRC --target-lang $TGT \
    --path ${RESULT_NAME}/checkpoint_best.pt \
    --beam 5 --lenpen 1.2 \
    --gen-subset $SET_TYPE \
    --remove-bpe=sentencepiece
fi