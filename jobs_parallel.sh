#!/bin/bash

#Author: Shufeng KONG, Cornell University, USA
#Contact: sk2299@cornell.edu
#
# This is an example script to run jobs in parallel. You can run the 69 jobs in multiple GPUs.
# Because the assignment of jobs depands on the memory size and the number of GPUs, you may need to revise this script
# to make it works. For example, we have 3 RTX 2080 GPUs, where each can run 8 jobs at a time. So we divide the
# 69 jobs into 9 groups and each GPU takes a group to run.
# If you want to run jobs one by one, you can use the script "jobs.py". I hope you like waiting :)


mkdir -p logs

model="run_HCLMP.py"
data_path="data/uvis_dataset_no_redundancy/uvis_dict.chkpt"
#data_path = 'data/uvis_dataset_no_gt/uvis_dict.chkpt'

train=0 # 0 for testing, 1 for training

transfer_type="gen_feat" # choices ['gen_feat', 'None']
#transfer_type="None"
epochs=40

val_dir="data/uvis_dataset_no_redundancy/idx/val_from_train/"
test_dir="data/uvis_dataset_no_redundancy/idx/test/"

train_dir="data/uvis_dataset_no_redundancy/idx/train_multiple/1/"
system_files=`ls ${train_dir}`

for fff in ${system_files[@]}
do
    train_path=${train_dir}/${fff}
    val_path=${val_dir}/${fff}
    test_path=${test_dir}/${fff}

    gpuid=1
    if [ ${train} -eq 1 ]
    then
        CUDA_VISIBLE_DEVICES=${gpuid} setsid python ${model} \
                                                      --train \
                                                      --epochs ${epochs} \
                                                      --transfer-type ${transfer_type} \
                                                      --data-path ${data_path} \
                                                      --train-path ${train_path} \
                                                      --val-path ${val_path} 1>logs/${fff}.log 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=${gpuid} setsid python ${model} \
                                                    --evaluate \
                                                    --epochs ${epochs} \
                                                    --transfer-type ${transfer_type} \
                                                    --data-path ${data_path} \
                                                    --test-path ${test_path} 1>logs/${fff}.log 2>&1 &
    fi
    echo "start process ${fff}"
done

#----------------------------------------------------------------------------------------------------------

train_dir="data/uvis_dataset_no_redundancy/idx/train_multiple/2/"
system_files=`ls ${train_dir}`

for fff in ${system_files[@]}
do
    train_path=${train_dir}/${fff}
    val_path=${val_dir}/${fff}
    test_path=${test_dir}/${fff}

    gpuid=2
    if [ ${train} -eq 1 ]
    then
        CUDA_VISIBLE_DEVICES=${gpuid} setsid python ${model} \
                                                      --train \
                                                      --epochs ${epochs} \
                                                      --transfer-type ${transfer_type} \
                                                      --data-path ${data_path} \
                                                      --train-path ${train_path} \
                                                      --val-path ${val_path} 1>logs/${fff}.log 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=${gpuid} setsid python ${model} \
                                                    --evaluate \
                                                    --epochs ${epochs} \
                                                    --transfer-type ${transfer_type} \
                                                    --data-path ${data_path} \
                                                    --test-path ${test_path} 1>logs/${fff}.log 2>&1 &
    fi
    echo "start process ${fff}"
done

#----------------------------------------------------------------------------------------------------------

train_dir="data/uvis_dataset_no_redundancy/idx/train_multiple/3/"
system_files=`ls ${train_dir}`

for fff in ${system_files[@]}
do
    train_path=${train_dir}/${fff}
    val_path=${val_dir}/${fff}
    test_path=${test_dir}/${fff}

    gpuid=3
    if [ ${train} -eq 1 ]
    then
        CUDA_VISIBLE_DEVICES=${gpuid} setsid python ${model} \
                                                      --train \
                                                      --epochs ${epochs} \
                                                      --transfer-type ${transfer_type} \
                                                      --data-path ${data_path} \
                                                      --train-path ${train_path} \
                                                      --val-path ${val_path} 1>logs/${fff}.log 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=${gpuid} setsid python ${model} \
                                                    --evaluate \
                                                    --epochs ${epochs} \
                                                    --transfer-type ${transfer_type} \
                                                    --data-path ${data_path} \
                                                    --test-path ${test_path} 1>logs/${fff}.log 2>&1 &
    fi
    echo "start process ${fff}"
done
echo "first wait!"
wait
#----------------------------------------------------------------------------------------------------------

train_dir="data/uvis_dataset_no_redundancy/idx/train_multiple/4/"
system_files=`ls ${train_dir}`

for fff in ${system_files[@]}
do
    train_path=${train_dir}/${fff}
    val_path=${val_dir}/${fff}
    test_path=${test_dir}/${fff}

    gpuid=1
    if [ ${train} -eq 1 ]
    then
        CUDA_VISIBLE_DEVICES=${gpuid} setsid python ${model} \
                                                      --train \
                                                      --epochs ${epochs} \
                                                      --transfer-type ${transfer_type} \
                                                      --data-path ${data_path} \
                                                      --train-path ${train_path} \
                                                      --val-path ${val_path} 1>logs/${fff}.log 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=${gpuid} setsid python ${model} \
                                                    --evaluate \
                                                    --epochs ${epochs} \
                                                    --transfer-type ${transfer_type} \
                                                    --data-path ${data_path} \
                                                    --test-path ${test_path} 1>logs/${fff}.log 2>&1 &
    fi
    echo "start process ${fff}"
done

#----------------------------------------------------------------------------------------------------------

train_dir="data/uvis_dataset_no_redundancy/idx/train_multiple/5/"
system_files=`ls ${train_dir}`

for fff in ${system_files[@]}
do
    train_path=${train_dir}/${fff}
    val_path=${val_dir}/${fff}
    test_path=${test_dir}/${fff}

    gpuid=2
    if [ ${train} -eq 1 ]
    then
        CUDA_VISIBLE_DEVICES=${gpuid} setsid python ${model} \
                                                      --train \
                                                      --epochs ${epochs} \
                                                      --transfer-type ${transfer_type} \
                                                      --data-path ${data_path} \
                                                      --train-path ${train_path} \
                                                      --val-path ${val_path} 1>logs/${fff}.log 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=${gpuid} setsid python ${model} \
                                                    --evaluate \
                                                    --epochs ${epochs} \
                                                    --transfer-type ${transfer_type} \
                                                    --data-path ${data_path} \
                                                    --test-path ${test_path} 1>logs/${fff}.log 2>&1 &
    fi
    echo "start process ${fff}"
done

#----------------------------------------------------------------------------------------------------------

train_dir="data/uvis_dataset_no_redundancy/idx/train_multiple/6/"
system_files=`ls ${train_dir}`

for fff in ${system_files[@]}
do
    train_path=${train_dir}/${fff}
    val_path=${val_dir}/${fff}
    test_path=${test_dir}/${fff}

    gpuid=3
    if [ ${train} -eq 1 ]
    then
        CUDA_VISIBLE_DEVICES=${gpuid} setsid python ${model} \
                                                      --train \
                                                      --epochs ${epochs} \
                                                      --transfer-type ${transfer_type} \
                                                      --data-path ${data_path} \
                                                      --train-path ${train_path} \
                                                      --val-path ${val_path} 1>logs/${fff}.log 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=${gpuid} setsid python ${model} \
                                                    --evaluate \
                                                    --epochs ${epochs} \
                                                    --transfer-type ${transfer_type} \
                                                    --data-path ${data_path} \
                                                    --test-path ${test_path} 1>logs/${fff}.log 2>&1 &
    fi
    echo "start process ${fff}"
done
echo "second wait!!"
wait
#----------------------------------------------------------------------------------------------------------

train_dir="data/uvis_dataset_no_redundancy/idx/train_multiple/7/"
system_files=`ls ${train_dir}`

for fff in ${system_files[@]}
do
    train_path=${train_dir}/${fff}
    val_path=${val_dir}/${fff}
    test_path=${test_dir}/${fff}

    gpuid=1
    if [ ${train} -eq 1 ]
    then
        CUDA_VISIBLE_DEVICES=${gpuid} setsid python ${model} \
                                                      --train \
                                                      --epochs ${epochs} \
                                                      --transfer-type ${transfer_type} \
                                                      --data-path ${data_path} \
                                                      --train-path ${train_path} \
                                                      --val-path ${val_path} 1>logs/${fff}.log 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=${gpuid} setsid python ${model} \
                                                    --evaluate \
                                                    --epochs ${epochs} \
                                                    --transfer-type ${transfer_type} \
                                                    --data-path ${data_path} \
                                                    --test-path ${test_path} 1>logs/${fff}.log 2>&1 &
    fi
    echo "start process ${fff}"
done

#----------------------------------------------------------------------------------------------------------


train_dir="data/uvis_dataset_no_redundancy/idx/train_multiple/8/"
system_files=`ls ${train_dir}`

for fff in ${system_files[@]}
do
    train_path=${train_dir}/${fff}
    val_path=${val_dir}/${fff}
    test_path=${test_dir}/${fff}

    gpuid=2
    if [ ${train} -eq 1 ]
    then
        CUDA_VISIBLE_DEVICES=${gpuid} setsid python ${model} \
                                                      --train \
                                                      --epochs ${epochs} \
                                                      --transfer-type ${transfer_type} \
                                                      --data-path ${data_path} \
                                                      --train-path ${train_path} \
                                                      --val-path ${val_path} 1>logs/${fff}.log 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=${gpuid} setsid python ${model} \
                                                    --evaluate \
                                                    --epochs ${epochs} \
                                                    --transfer-type ${transfer_type} \
                                                    --data-path ${data_path} \
                                                    --test-path ${test_path} 1>logs/${fff}.log 2>&1 &
    fi
    echo "start process ${fff}"
done

#----------------------------------------------------------------------------------------------------------


train_dir="data/uvis_dataset_no_redundancy/idx/train_multiple/9/"
system_files=`ls ${train_dir}`

for fff in ${system_files[@]}
do
    train_path=${train_dir}/${fff}
    val_path=${val_dir}/${fff}
    test_path=${test_dir}/${fff}

    gpuid=3
    if [ ${train} -eq 1 ]
    then
        CUDA_VISIBLE_DEVICES=${gpuid} setsid python ${model} \
                                                      --train \
                                                      --epochs ${epochs} \
                                                      --transfer-type ${transfer_type} \
                                                      --data-path ${data_path} \
                                                      --train-path ${train_path} \
                                                      --val-path ${val_path} 1>logs/${fff}.log 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=${gpuid} setsid python ${model} \
                                                    --evaluate \
                                                    --epochs ${epochs} \
                                                    --transfer-type ${transfer_type} \
                                                    --data-path ${data_path} \
                                                    --test-path ${test_path} 1>logs/${fff}.log 2>&1 &
    fi
    echo "start process ${fff}"
done
echo "third wait!!"
wait
echo "Done!!"
#----------------------------------------------------------------------------------------------------------


