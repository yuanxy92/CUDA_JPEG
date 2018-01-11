#include <fstream>
#include <opencv2/opencv.hpp>

#include "video_decoder.hpp"

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
	char* imgname = "E:\\Project\\CUDA_JPEG\\data\\0.jpg";
	char* jpegdata;
	int length;
	loadJpeg(imgname, jpegdata, length);
	char* jpeg_data_d;
	cudaMalloc(&jpeg_data_d, length);
	cudaMemcpy(jpeg_data_d, jpegdata, length, cudaMemcpyHostToDevice);

	cv::Mat img(3008, 4000, CV_8UC3);

	cv::Ptr<cv::cudacodec::detail::VideoDecoder> decoderPtr;

	CUVIDPARSERDISPINFO displayInfo;
	displayInfo.progressive_frame = 1;
	int num_fields = 1;

	CUVIDPROCPARAMS videoProcParams;
	std::memset(&videoProcParams, 0, sizeof(CUVIDPROCPARAMS));
	videoProcParams.progressive_frame = displayInfo.progressive_frame;
	videoProcParams.second_field = 1;
	videoProcParams.top_field_first = 1;
	videoProcParams.unpaired_field = (num_fields == 1);
	videoProcParams.raw_input_dptr = (unsigned long long)jpeg_data_d;
	videoProcParams.raw_output_pitch = 0;

	cv::cuda::GpuMat decodedFrame = decoderPtr->mapFrame(0, videoProcParams);


	return 0;
}