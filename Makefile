all: stream.txt runtime.txt csvs

stream.txt: stream.o
	./stream.o > stream.txt

stream.o: stream.c
	gcc -o stream.o stream.c

runtime.txt: uncompressed.h5

uncompressed.h5: overthrust_3D_initial_model.h5 zfp-0.5.3/lib/libzfp.so simple.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib DEVITO_OPENMP=1 python -u simple.py > runtime.txt 2>&1

csvs: precision.csv tolerance.csv rate.csv precision-s.csv tolerance-s.csv rate-s.csv

precision.csv: uncompressed.h5 precision.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib python -u precision.py uncompressed.h5 > precision.csv

tolerance.csv: uncompressed.h5 tolerance.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib python -u tolerance.py uncompressed.h5 > tolerance.csv

rate.csv: uncompressed.h5 rate.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib python -u rate.py uncompressed.h5 > rate.csv

precision-s.csv: uncompressed.h5 precision.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib python -u precision.py uncompressed.h5 --no-parallel > precision-s.csv

tolerance-s.csv: uncompressed.h5 tolerance.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib python -u tolerance.py uncompressed.h5 --no-parallel > tolerance-s.csv

rate-s.csv: uncompressed.h5 rate.py
	LD_LIBRARY_PATH=./zfp-0.5.3/lib python -u rate.py uncompressed.h5 --no-parallel > rate-s.csv
