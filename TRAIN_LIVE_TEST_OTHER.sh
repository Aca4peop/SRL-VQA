datasets=("CSIQ_VQA" "KON_VQA" "VQC_VQA")
frames=("16" "16" "16")
best=("0.8" "0.7" "0.7")

for ((i = 0; i < ${#datasets[@]}; i ++ ))
do
    echo `python3 cross_train_test.py --batch-size=4 --batch-test=1 --dataset=LIVE_VQA --dataset_test=${datasets[$i]} --frame=${frames[$i]} --fine_tune=True --base_lr=2e-4 --loss=mix --best=${best[$i]} --epoch=20`;
done

