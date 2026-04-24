for i in {0..9}
do
  dataset="100_$i"
  python moead_exp.py $dataset 
  python nsga_exp.py $dataset
done

for i in {0..9}
do
  dataset="150_$i"
  python moead_exp.py $dataset
  python nsga_exp.py $dataset
done

for i in {0..9}
do
  dataset="200_$i"
  python moead_exp.py $dataset
  python nsga_exp.py $dataset
done

for i in {0..9}
do
  dataset="250_$i"
  python moead_exp.py $dataset
  python nsga_exp.py $dataset
done