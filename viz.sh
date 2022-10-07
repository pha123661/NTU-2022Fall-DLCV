python3 P2_plot.py
# 3 models
for filename in "0013_sat" "0062_sat" "0104_sat"
do
    for folder in "first" "mid" "last"
    do
        python3 viz_mask.py --img_path="./p2 plot fig/$filename.jpg" --seg_path="./p2 plot fig/$folder/$filename.png"
        mv exp.png "./p2 plot fig/$folder/$filename rst.png"
    done
done

# ground truth
for filename in "0013" "0062" "0104"
do
    python3 viz_mask.py --img_path="./p2 plot fig/${filename}_sat.jpg" --seg_path="./p2 plot fig/${filename}_mask.png"
    mv exp.png "./p2 plot fig/$filename gt.png"
done