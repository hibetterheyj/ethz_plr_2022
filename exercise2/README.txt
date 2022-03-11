# To install dependencies
cd path/to/Exercise2
pip install -r requirements.txt

# To train the policy using the default settings
python cart_ctrl.py

# To train the policy and save it to a specified file
python cart_ctrl.py -train file_name.pickle

# To train the policy using only the cart position and pole angle as state
python cart_ctrl.py -non_markov

# To test a saved policy
python cart_ctrl.py -test file_name.pickle
