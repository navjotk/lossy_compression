all: errors images csvs

errors: error-decompressed-t-0.h5.txt error-decompressed-t-8.h5.txt error-decompressed-t-15.h5.txt error-decompressed-p-6.h5.txt error-decompressed-p-10.h5.txt error-decompressed-p-19.h5.txt

images: uncompressed.png decompressed-t-0.png decompressed-t-8.png decompressed-t-15.png decompressed-p-6.png decompressed-p-10.png decompressed-p-19.png

csvs: precision.csv tolerance.csv

errors-%.txt: % uncompressed.h5 difference.py
	python -u difference.py uncompressed.h5 $< | tee $@

%.png: %.h5
	python plot_numpy_hdf5.py $< $@

precision.csv: uncompressed.h5 precision.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib python precision.py uncompressed.h5 > precision.csv

tolerance.csv: uncompressed.h5 tolerance.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib python tolerance.py uncompressed.h5 > tolerance.csv

uncompressed.h5: overthrust_3D_initial_model.h5 zfp-0.5.3/lib/libzfp.so simple.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib DEVITO_OPENMP=1 python simple.py

zfp-0.5.3/lib/libzfp.so: zfp-0.5.3.tar.gz
	tar -xzvf zfp-0.5.3.tar.gz
	cd zfp-0.5.3 && make && make shared

overthrust_3D_initial_model.h5:
	wget ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/3DFWI/overthrust_3D_initial_model.h5

zfp-0.5.3.tar.gz: 
	wget https://computation.llnl.gov/projects/floating-point-compression/download/zfp-0.5.3.tar.gz

