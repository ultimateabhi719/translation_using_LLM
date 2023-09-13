#!/usr/bin/env fish


# smallTrain
python src/translator.v1.py  train --device cuda:0 --maxlen 100 \
                                                --subset 25000 \
                                                --batch_size 72 \
                                                --num_epochs 200 \
                                                --save_prefix runs/en_de/smallTrain \
                                                --resume_dir None \
                                                --scheduler_freq 200

# FullTrain
python src/translator.v1.py  train --device cuda:0 --maxlen 50 \
                                                --subset None \
                                                --batch_size 72 \
                                                --num_epochs 20 \
                                                --save_prefix runs/en_de/log0 \
                                                --resume_dir runs/en_de/smallTrain_maxlen100_subset25k/ \
                                                --scheduler_freq 200