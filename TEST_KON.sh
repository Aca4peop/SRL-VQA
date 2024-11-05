j=5

# Fine-tuning

for ((i=0; i<j; i++))
do
    echo `python3 train_vqa_yuv.py --idx=$i --batch-size=16 --batch-test=8 --dataset=KON_VQA --frame=16 --base_lr=2e-4 --loss=plcc --fine_tune=True --best=0.84`;
done

## Baseline

# for ((i=0; i<j; i++))
# do
#     echo `python3 train_vqa_yuv.py --idx=$i --batch-size=16 --batch-test=8 --dataset=KON_VQA --frame=16 --base_lr=2e-4 --loss=plcc --best=0.84`;
# done