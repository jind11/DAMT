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
  --src_data_name)
    SRC_DATA_NAME="$2"; shift 2;;
  --reload_checkpoint)
    RELOAD_CHECKPOINT="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"


python -W ignore train.py \
    --exp_name $DATA_NAME\_$SRC\_$TGT\_${SRC_DATA_NAME} \
    --dump_path ./eval/ \
    --data_path /data/medg/misc/jindi/nlp/datasets/OPUS/$SRC-$TGT/$DATA_NAME/processed/$SRC-$TGT \
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
    --tokens_per_batch 1500 \
    --batch_size 16 \
    --bptt 256 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
    --epoch_size 10 \
    --eval_bleu true \
    --stopping_criterion valid_$SRC-$TGT\_mt_bleu,10 \
    --validation_metrics valid_$SRC-$TGT\_mt_bleu \
    --max_epoch 1 \
    --reload_checkpoint $RELOAD_CHECKPOINT \
    --max_len 150 \
    --eval_only true \