1)my_train.py 前向传播
2)my_render.py
3)arguments/init、gaussian_renderer/init、scene/init、camera、gaussian_model等等

1)只增加ATF模块
python atf_train.py -s ../data/TI-NSD/apples -m ../output/ti-nsd/atf/apples --eval
python atf_render.py -m ../output/ti-nsd/atf/apples -s ../data/TI-NSD/apples --skip_train
python metrics.py -m ../output/ti-nsd/atf/apples