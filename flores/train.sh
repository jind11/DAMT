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

fairseq-train \
/home/ubuntu/proj/data/$ORDERED_SRC-$ORDERED_TGT/data_bin/ \
--source-lang $SRC --target-lang $TGT \
--arch transformer --share-all-embeddings \
--encoder-layers 5 --decoder-layers 5 \
--encoder-embed-dim 512 --decoder-embed-dim 512 \
--encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
--encoder-attention-heads 8 --decoder-attention-heads 8 \
--encoder-normalize-before --decoder-normalize-before \
--dropout 0.3 --attention-dropout 0.3 --relu-dropout 0.3 \
--weight-decay 0.0001 \
--label-smoothing 0.2 --criterion label_smoothed_cross_entropy \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
--lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 \
--lr 3e-3 --min-lr 1e-9 \
--max-tokens 8000 \
--update-freq 4 \
--max-epoch 100 --save-interval 5 \
--no-epoch-checkpoints \
--log-format json \
--log-interval 100 \
--ddp-backend no_c10d \
--seed 1 \
--save-dir tmp/sup_${SRC}_${TGT}_new