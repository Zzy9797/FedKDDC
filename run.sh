dataset="cifar10"
type='mode'
skew_class=2
iternum=200
beta=0.1
alpha1=0.8
alpha2=2.5
alpha3=0.5
T=0.8


# Ours
python train.py --gpu "7" --dataset $dataset --type $type --skew_class $skew_class --num_local_iterations $iternum --beta $beta --alpha1 $alpha1 --alpha2 $alpha2 --alpha3 $alpha3 --T $T 


    