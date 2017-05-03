
histogram_demo: histogram_demo.cpp histogram.h histogram_storage.h
	$(CXX) -std=c++1y histogram_demo.cpp -o histogram_demo -lhdf5