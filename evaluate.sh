#/bin/bash

light_condition_dir=~/workspace/dataset/2ndEXP/stage2/C2-LightCondition/videos
dark_condition_dir=~/workspace/dataset/2ndEXP/stage2/C1-DarkCondition/videos
dark_light_condition_dir=~/workspace/dataset/2ndEXP/stage2/C3-LightDarkCondition/videos

result_dir=result

# for f in $light_condition_dir/*.mp4
# do
#     echo "Processing $f"
#     if [ -e $result_dir/"${f##*/}"_timestamp.csv ]
#     then
#         echo "exist! == skip"
#     else
#         echo "Detecting $f"
#         time python detect.py --condition light --weights ~/workspace/weights/c2-l-sg2-50-ep.pt --conf 0.35 --source $f --csv-folder result/ --project result/

#     fi
# done

# result_dir=result/dark

# for f in $dark_condition_dir/*.mp4
# do
#     echo "Processing $f"
#     if [ -e $result_dir/"${f##*/}"_timestamp.csv ]
#     then
#         echo "exist! == skip"
#     else
#         echo "Detecting $f"
#         time python detect.py --condition dark --weights ~/workspace/weights/c1-d-sg2-50-ep.pt --conf 0.35 --source $f --csv-folder result/dark/ --project result/dark/

#     fi
# done

result_dir=result/stage2/darklight

for f in $dark_light_condition_dir/*.mp4
do
    echo "Processing $f"
    if [ -e $result_dir/"${f##*/}"_timestamp.csv ]
    then
        echo "exist! == skip"
    else
        echo "Detecting $f"
        time python detect.py --condition darklight --weights ~/workspace/weights/c1-d-sg2-50-ep.pt \
            --dark-condition-weights ~/workspace/weights/c1-d-sg2-50-ep.pt \
            --light-condition-weights ~/workspace/weights/c2-l-sg2-50-ep.pt \
            --stage 2 \
            --view-img \
            --conf 0.35 --source $f --csv-folder result/stage2/darklight/ --project result/stage2/darklight/

    fi
done

# find $light_condition_dir -name "*.mp4" -exec time python detect.py --source {} --condition light --conf-thres 0.35  --weight ~/workspace/weights/c2-l-sg2-50-ep.pt --csv-folder result/ --project result/ \;