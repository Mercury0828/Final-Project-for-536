# CUDA_VISIBLE_DEVICES=0 bash run.sh center_single 2>&1

# center_single
# distribute_four
# left_center_single_right_center_single
# in_distribute_four_out_center_single

declare -a models=("cnn" "resnet" "lstm" "wren" "efficientnet")

args=$1

for model in "${models[@]}"; do
    echo $model
    python main.py --dataset i_raven --validate_interval 1 --verbose 1 --batch_size 100 --lr 1e-3 --epochs 300 --tensorboard --model $model --configure $1 
done
