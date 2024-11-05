j=5

## Fine-tuning

for ((i=0; i<j; i++))
do
    echo `python3 train_vqa_type.py --batch-size=8 --batch-test=8 --dataset=CSIQ_VQA --frame=32 --base_lr=2e-4 --loss=plcc --fine_tune=True --best=0.92 --idx=$i`;
done


## Baseline

# for ((i=0; i<j; i++))
# do
#     echo `python3 train_vqa_type.py --batch-size=8 --batch-test=8 --dataset=CSIQ_VQA --frame=32 --base_lr=2e-4 --loss=plcc --best=0.92 --idx=$i`;
# done
