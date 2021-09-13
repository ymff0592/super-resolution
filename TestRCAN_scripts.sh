#!/bin/bash/

# x2
CUDA_VISIBLE_DEVICES=0 python main.py --data_test Set5 --scale 2 --model LATTICENET  --n_feats 64 --pre_train ../premodel/LatticeNet_x2.pt --test_only --save_results  --save 'latticenet2x' --testpath .../dataset --testset Set5
CUDA_VISIBLE_DEVICES=0 python main.py --data_test Set14 --scale 2 --model LATTICENET --n_feats 64  --pre_train ../premodel/LatticeNet_x2.pt --test_only --save_results --save 'latticenet2x' --testpath .../dataset --testset Set14
CUDA_VISIBLE_DEVICES=0 python main.py --data_test B100 --scale 2 --model LATTICENET  --n_feats 64 --pre_train ../premodel/LatticeNet_x2.pt --test_only --save_results  --save 'latticenet2x' --testpath .../dataset --testset B100
CUDA_VISIBLE_DEVICES=0 python main.py --data_test Urban100 --scale 2 --model LATTICENET --n_feats 64 --pre_train ../premodel/LatticeNet_x2.pt --test_only --save_results  --save 'latticenet2x' --testpath .../dataset --testset Urban100

# x3
CUDA_VISIBLE_DEVICES=0 python main.py --data_test Set5 --scale 3 --model LATTICENET --n_feats 64 --pre_train ../premodel/LatticeNet_x3.pt --test_only --save_results  --save 'latticenet3x' --testpath .../dataset --testset Set5
CUDA_VISIBLE_DEVICES=0 python main.py --data_test Set14 --scale 3 --model LATTICENET --n_feats 64 --pre_train ../premodel/LatticeNet_x3.pt --test_only --save_results --save 'latticenet3x' --testpath .../dataset --testset Set14
CUDA_VISIBLE_DEVICES=0 python main.py --data_test B100 --scale 3 --model LATTICENET --n_feats 64 --pre_train ../premodel/LatticeNet_x3.pt --test_only --save_results  --save 'latticenet3x' --testpath .../dataset --testset B100
CUDA_VISIBLE_DEVICES=0 python main.py --data_test Urban100 --scale 3 --model LATTICENET --n_feats 64 --pre_train ../premodel/LatticeNet_x3.pt --test_only --save_results  --save 'latticenet3x' --testpath .../dataset --testset Urban100

# x4
CUDA_VISIBLE_DEVICES=0 python main.py --data_test Set5 --scale 4 --model LATTICENET --n_feats 64 --pre_train ../premodel/LatticeNet_x4.pt --test_only --save_results  --save 'latticenet4x' --testpath .../dataset --testset Set5
CUDA_VISIBLE_DEVICES=0 python main.py --data_test Set14 --scale 4 --model LATTICENET --n_feats 64 --pre_train ../premodel/LatticeNet_x4.pt --test_only --save_results --save 'latticenet4x' --testpath .../dataset --testset Set14
CUDA_VISIBLE_DEVICES=0 python main.py --data_test B100 --scale 4 --model LATTICENET  --n_feats 64 --pre_train ../premodel/LatticeNet_x4.pt --test_only --save_results  --save 'latticenet4x' --testpath .../dataset --testset B100
CUDA_VISIBLE_DEVICES=0 python main.py --data_test Urban100 --scale 4 --model LATTICENET --n_feats 64 --pre_train ../premodel/LatticeNet_x4.pt --test_only --save_results  --save 'latticenet4x' --testpath .../dataset --testset Urban100

