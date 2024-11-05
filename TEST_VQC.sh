j=5

## Fine-tuning
for ((i=0; i<j; i++))
do
    echo `python3 train_vqa_yuv.py --batch-size=16 --batch-test=1 --dataset=VQC_VQA --frame=16 --base_lr=2e-4 --fine_tune=True --loss=plcc --best=0.74 --idx=$i`;
done

## Baseline
# for ((i=0; i<j; i++))
# do
#     echo `python3 train_vqa_yuv.py --batch-size=16 --batch-test=1 --dataset=VQC_VQA --frame=16 --base_lr=2e-4 --loss=plcc --best=0.74 --idx=$i`;
# done