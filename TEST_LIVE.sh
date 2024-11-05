j=5

## Fine-tuning

for ((i=0; i<j; i++))
do
    echo `python3 train_vqa_type.py --batch-size=8 --batch-test=4 --dataset=LIVE_VQA --frame=32 --base_lr=2e-4 --fine_tune=True --loss=plcc --best=0.9 --idx=$i`;
done

## Baseline

# for ((i=0; i<j; i++))
# do
#     echo `python3 train_vqa_type.py --batch-size=8 --batch-test=4 --dataset=LIVE_VQA --frame=32 --base_lr=2e-4 --loss=plcc --best=0.9 --idx=$i`;
# done