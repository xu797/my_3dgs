cd /home/eric/xujixian/work/code/gaussian-splatting
conda activate gs

# 0) 可选：检查依赖
command -v ffmpeg
command -v colmap

# 1) 从视频抽帧到 input（convert.py 需要 input 目录）
mkdir -p data/ggbond/input
ffmpeg -i data/ggbond/ggbond.mp4 -vf "fps=2" -q:v 2 data/ggbond/input/%05d.jpg

# 2) 跑 COLMAP + undistort，生成 sparse/0 与 images
python convert.py -s data/ggbond


# 3) 开始训练（输出到 data/ggbond/output）
python train.py -s data/ggbond -m data/ggbond/output

综上：input是视频抽帧，output是3dgs输出，其余都是colmap的结果。

执行以下命令 每8张图片选取一个测试图片 必须加上--eval 不然没有测试集 最后不能使用metrics.py进行评估。
python train.py -s data/ggbond -m data/ggbond/output --eval

#--skip_train 表示只渲染出test的图片
python render.py -m data/ggbond/output -s data/ggbond --skip_train 

#评估
python metrics.py -m data/ggbond/output

TI-NSD
python train.py -s data/TI-NSD/apples -m output/ti-nsd/apples --eval
python render.py -m output/ti-nsd/apples -s data/TI-NSD/apples --skip_train
python metrics.py -m output/ti-nsd/apples