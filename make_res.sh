#!/bin/bash
orig_dir="illustration/test_orig"
for train_loop in identity simple cycle harmonic
do
  if [ "$train_loop" == simple ]
  then
    loop_token="-s"
  elif [ "$train_loop" == cycle ]
  then
    loop_token="-c"
  elif [ "$train_loop" == harmonic ]
  then
    loop_token="-l"
  elif [ "$train_loop" == identity ]
  then
    loop_token=""
  else
    echo "bug in train_loop switch"
    exit 1
  fi

  for model in pix dense
  do
    if [ "$model" == pix ]
    then
      model_token=""
    elif [ "$model" == dense ]
    then
      model_token="-d"
    else
      echo "bug in model switch"
      exit 1
    fi

    for pretrain in pretrain nopretrain
    do
      if [ "$pretrain" == pretrain ]
      then
        pretrain_token="-p"
      elif [ "$pretrain" == nopretrain ]
      then
        pretrain_token=""
      else
        echo "bug in pretrain switch"
        exit 1
      fi

      for mode in train eval
      do
        if [ "$mode" == train ]
        then
          dropout_token="0.5"
        elif [ "$mode" == eval ]
        then
          dropout_token="0.0"
        else
          echo "bug in mode switch"
          exit 1
        fi

        for file in $(ls "$orig_dir"/*png)
        do
          # mkdir -p illustration/"$train_loop"/"$model"/"$pretrain"/"$mode"
          # in_file="$orig_dir"/"$file"
          in_file="$file"
          out_file=illustration/"$train_loop"/"$model"/"$pretrain"/"$mode"/$(basename "$file")
          # echo ./main.py "$loop_token" "$model_token" "$pretrain_token" -D "$dropout_token" -i "$in_file" -o "$out_file"
          ./main.py $loop_token $model_token $pretrain_token -D $dropout_token -i $in_file -o $out_file
        done
      done
    done
  done
done
