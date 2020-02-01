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
  --train_type)
    TRAIN_TYPE="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"

epoch_size=$(cat /data/medg/misc/jindi/nlp/datasets/OPUS/$SRC-$TGT/$TGT_DATA_NAME/${TRAIN_TYPE}_back_translate/$SRC_DATA_NAME/train.$SRC-$TGT.$SRC | wc -l)
epoch_size=$((epoch_size/2))
echo $epoch_size

if [ $TGT_DATA_NAME == "KORAN" ]; then
    max_epoch=10
elif [ $TGT_DATA_NAME == "EMEA" ]; then
    max_epoch=10
elif [ $TGT_DATA_NAME == "IT" ]; then
    max_epoch=10
else
    max_epoch=10
fi
echo $max_epoch

python -W ignore train.py \
    --exp_name no_pretrain_sup_${TRAIN_TYPE}_back_src_$SRC_DATA_NAME\_tgt_$TGT_DATA_NAME\_$SRC\_$TGT \
    --dump_path ./tmp/ \
    --data_path /data/medg/misc/jindi/nlp/datasets/OPUS/$SRC-$TGT/$TGT_DATA_NAME/processed/$SRC-$TGT \
    --para_data_path /data/medg/misc/jindi/nlp/datasets/OPUS/$SRC-$TGT/$TGT_DATA_NAME/${TRAIN_TYPE}_back_translate/$SRC_DATA_NAME \
    --lgs $SRC-$TGT \
    --mt_steps $SRC-$TGT \
    --encoder_only false \
    --emb_dim 1024 \
    --n_layers 6 \
    --n_heads 8 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --gelu_activation true \
    --tokens_per_batch 2500 \
    --batch_size 32 \
    --bptt 256 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
    --epoch_size $epoch_size \
    --eval_bleu true \
    --stopping_criterion valid_$SRC-$TGT\_mt_bleu,5 \
    --validation_metrics valid_$SRC-$TGT\_mt_bleu \
    --max_epoch $max_epoch \
    --max_len 150 \