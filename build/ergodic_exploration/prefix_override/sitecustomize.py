import sys
if sys.prefix == '/home/avavvas/venvs/venv':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/avavvas/dipl/install/ergodic_exploration'
