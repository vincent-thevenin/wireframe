V1_LINK="https://drive.google.com/uc?export=download&id=1wiZlXPUUyi3-SFqWBRMRVzE8Jt_-5sEQ"

POINTLINES_LINK="https://drive.google.com/uc?export=download&id=1ZTuCG9R_C9aD1WVN7AC_hwi8yTRS-qyg"

LINEMAT_LINK="https://drive.google.com/uc?export=download&id=1HvlQGhMHeviTCFJmO5PmxpFpcCGObtpX"

all: build

build:
	
	wget ${V1_LINK} -O v1.1.zip
	
	wget ${POINTLINES_LINK} -O pointlines.zip
	
	wget ${LINEMAT_LINK} -O line_mat.zip
	
	unzip v1.1.zip
	unzip pointlines.zip
	unzip line_mat.zip
	
	mkdir data
	mv v1.1 data/v1.1
	mv pointlines data/pointlines
	mv line_mat evaluation/wireframe/line_mat
	
	cd junc
	mkdir ../data/junc
	mkdir ../data/junc/processed
	python3 main.py --create_dataset --exp 1 --json
	cd ..
	
	cd linepx
	mkdir ../result
	mkdir ../result/linepx
	python3 main.py --genLine
	
	cd ../junc
	mkdir ../logs
	python3 main.py --exp 1 --json --gpu 0 --balance
	
	cd ../linepx/
	python3 main.py --netType stackedHGB --GPUs 0 --LR 0.001 --batchSize 4
	
	
