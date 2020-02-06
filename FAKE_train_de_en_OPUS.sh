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
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"

if [ "$SRC" != 'en' ]; then
    OTHER_LANG=$SRC
else
    OTHER_LANG=$TGT
fi
echo $OTHER_LANG

if [ "$SRC" \< "$TGT" ]; then
    ORDERED_SRC=$SRC
    ORDERED_TGT=$TGT
else
    ORDERED_SRC=$TGT
    ORDERED_TGT=$SRC
fi

pretrained_model_dir=/home/ubuntu/proj/models/XLM
data_dir=/home/ubuntu/proj/data/$ORDERED_SRC-$ORDERED_TGT

epoch_size=$(cat ${data_dir}/$DATA_NAME/processed/$ORDERED_SRC-$ORDERED_TGT/train.$SRC | wc -l)
max_epoch_size=300000
epoch_size=$((epoch_size>max_epoch_size ?  max_epoch_size : epoch_size))
echo $epoch_size

#pretrained_model_dir=/data/medg/misc/jindi/nlp/embeddings/XLM
#data_dir=/data/medg/misc/jindi/nlp/datasets/OPUS/$ORDERED_SRC-$ORDERED_TGT


python -W ignore train.py \
    --exp_name umt_$DATA_NAME\_$SRC\_$TGT \
    --dump_path ./tmp/ \
    --data_path ${data_dir}/$DATA_NAME/processed/$ORDERED_SRC-$ORDERED_TGT \
    --lgs $SRC-$TGT \
    --ae_steps $SRC,$TGT \
    --bt_steps $SRC-$TGT-$SRC,$TGT-$SRC-$TGT \
    --word_shuffle 3 \
    --word_dropout 0.1 \
    --word_blank 0.1 \
    --lambda_ae '0:1,100000:0.1,300000:0' \
    --encoder_only false \
    --emb_dim 1024 \
    --n_layers 6 \
    --n_heads 8 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --gelu_activation true \
    --tokens_per_batch 100 \
    --batch_size 10 \
    --bptt 256 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
    --epoch_size $epoch_size \
    --eval_bleu true \
    --validation_metrics valid_$SRC-$TGT\_mt_bleu \
    --max_epoch 10000 \
    --max_len 150 \