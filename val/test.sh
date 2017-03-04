#!/bin/bash
#metric
python metric.py ../../mq_data/groundtruth/trumper.gt2 ../../mq_result/log/0302/white_old_none.log


#svd
python metric.py ../../mq_data/groundtruth/trumper.gt2 ../../mq_result/log/0302/svd/white_old_svd_origin.log
python metric.py ../../mq_data/groundtruth/trumper.gt2 ../../mq_result/log/0302/svd/white_old_svd_acc.log
python metric.py ../../mq_data/groundtruth/trumper.gt2 ../../mq_result/log/0302/svd/svd_origin_en.log

#knn
python metric.py ../../mq_data/groundtruth/trumper.gt2 ../../mq_result/log/0302/knn/white_old_knn.log
python metric.py ../../mq_data/groundtruth/trumper.gt2 ../../mq_result/log/0302/knn/white_old_knn_acc.log
python metric.py ../../mq_data/groundtruth/trumper.gt2 ../../mq_result/log/0302/knn/knn_origin_en.log

#knntext
python metric.py ../../mq_data/groundtruth/trumper.gt2 ../../mq_result/log/0302/knntext/white_old_origin.log
python metric.py ../../mq_data/groundtruth/trumper.gt2 ../../mq_result/log/0302/knntext/white_old_knntext_origin_acc.log
python metric.py ../../mq_data/groundtruth/trumper.gt2 ../../mq_result/log/0302/knntext/knntext_origin_en.log


#rouge
python rouge.py ../../mq_data/groundtruth/trumper.gt2 ../../mq_result/log/0302/white_old_none.log

#svd
python rouge.py ../../mq_data/groundtruth/trumper.gt2 ../../mq_result/log/0302/svd/white_old_svd_origin.log
python rouge.py ../../mq_data/groundtruth/trumper.gt2 ../../mq_result/log/0302/svd/white_old_svd_acc.log
python rouge.py ../../mq_data/groundtruth/trumper.gt2 ../../mq_result/log/0302/svd/svd_origin_en.log

#knn
python rouge.py ../../mq_data/groundtruth/trumper.gt2 ../../mq_result/log/0302/knn/white_old_knn.log
python rouge.py ../../mq_data/groundtruth/trumper.gt2 ../../mq_result/log/0302/knn/white_old_knn_acc.log
python rouge.py ../../mq_data/groundtruth/trumper.gt2 ../../mq_result/log/0302/knn/knn_origin_en.log

#knntext
python rouge.py ../../mq_data/groundtruth/trumper.gt2 ../../mq_result/log/0302/knntext/white_old_origin.log
python rouge.py ../../mq_data/groundtruth/trumper.gt2 ../../mq_result/log/0302/knntext/white_old_knntext_origin_acc.log
python rouge.py ../../mq_data/groundtruth/trumper.gt2 ../../mq_result/log/0302/knntext/knntext_origin_en.log
