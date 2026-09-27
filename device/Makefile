# Build lib/libgsv_qnn.so natively on the board (Ubuntu 22.04+ aarch64, needs g++ and make).
# A cross-compiled lib/libgsv_qnn.so is already included; rebuild only if needed.
QNN_INCLUDE ?= include/QNN
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -fPIC -Wall -Wno-missing-field-initializers

lib/libgsv_qnn.so: src/gsv_qnn.cpp
	@mkdir -p lib
	$(CXX) $(CXXFLAGS) -shared -I$(QNN_INCLUDE) $< -o $@ -ldl

clean:
	rm -f lib/libgsv_qnn.so

.PHONY: clean
