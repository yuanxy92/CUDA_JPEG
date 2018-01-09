/**
@brief demo for npp jpeg coder
@author Shane Yuan
@date Jan 09, 2018
*/
#include <cstdio>
#include <cstdlib>

#include "NPPJpegCoder.h"

int main(int argc, char* argv[]) {
	std::vector<npp::NPPJpegCoder> coder(8);
	std::vector<char*> string(8);
	std::vector<size_t> fsize(8);
	std::vector<cv::cuda::GpuMat> img_d(8);
	std::vector<cv::Mat> img(8);

	for (int i = 0; i < 8; i++) {
		char imgname[256];
		sprintf(imgname, "E:\\Project\\CUDA_JPEG\\data\\%d.jpg", i);
		FILE *f = fopen(imgname, "rb");
		fseek(f, 0, SEEK_END);
		fsize[i] = ftell(f);
		fseek(f, 0, SEEK_SET);
		string[i] = (char *)malloc(fsize[i] + 1);
		fread(string[i], fsize[i], 1, f);
		fclose(f);
	}

	for (size_t i = 0; i < 8; i++) {
		coder[i].init(4000, 3000, 85);
		img_d[i].create(3000, 4000, CV_8UC3);
	}
	for (size_t i = 0; i < 8; i++) {
		coder[i].decode((uchar*)string[i], fsize[i], img_d[i], 0);
	}

	for (size_t i = 0; i < 8; i++)
		img_d[i].download(img[i]);

	return 0;
}