#!/bin/bash

python3 cnn_pytorch.py > training_log.txt

chmod +x f1.sh precision_recall.sh

bash f1.sh
bash precision_recall.sh
