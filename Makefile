all: precision.csv tolerance.csv uncompressed.png decompressed-t-0.png errors.txt decompressed-t-8.png decompressed-t-15.png

errors.txt: uncompressed.h5 decompressed-t-0.h5 difference.py
	python difference.py uncompressed.h5 decompressed-t-0.h5 > errors.txt

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

