ModuleNotFoundError: No module named 'pygame'
python -m pip install pygame

python cart_ctrl_yujie.py -train part1_1

python cart_ctrl_yujie.py -non_markov -train part2_1

# with different reduced state
python cart_ctrl_yujie.py -non_markov -use_diff -train part3_1
