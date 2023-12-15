c:
	clang++ add.cpp -o add

cuda:
	nvcc add.cu -o add_cuda
