#include <nvpipe.h>
#include <fstream>

#include <opencv2/opencv.hpp>

void loadJpeg(const char *input_file, char *&pJpegData, int &nInputLength) {
	// Load file into CPU memory
	std::ifstream stream(input_file, std::ifstream::binary);

	if (!stream.good())
	{
		return;
	}

	stream.seekg(0, std::ios::end);
	nInputLength = (int)stream.tellg();
	stream.seekg(0, std::ios::beg);

	pJpegData = new char[nInputLength];
	stream.read(reinterpret_cast<char *>(pJpegData), nInputLength);
}

int main(int argc, char* argv[]) {


	char* name = "E:\\Project\\CUDA_JPEG\\data\\0.jpg";
	char* jpegdata;
	int length;
	loadJpeg(name, jpegdata, length);

	nvpipe* dec = nvpipe_create_decoder(NVPIPE_H264_NV);

	size_t width = 4000, height = 3000;

	cv::Mat img(3000, 4000, CV_8UC3);
	nvp_err_t status = nvpipe_decode(dec, jpegdata, length, img.data, img.cols, img.rows, NVPIPE_RGB);

	nvpipe_destroy(dec); // destroy the decoder when finished
	return 0;
}