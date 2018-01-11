#include <fstream>
#include <opencv2/opencv.hpp>

#include <npp.h>
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

int main2(int argc, char* argv[]) {
	char* imgname = "E:\\Project\\CUDA_JPEG\\data\\0.jpg";
	char* jpegdata;
	int length;
	loadJpeg(imgname, jpegdata, length);
	char* jpeg_data_d;
	cudaMalloc(&jpeg_data_d, length);
	cudaMemcpy(jpeg_data_d, jpegdata, length, cudaMemcpyHostToDevice);

	cv::Mat img(3008, 4000, CV_8UC3);
	CUvideoctxlock lock(0);
	cv::Ptr<cv::cudacodec::detail::VideoDecoder> decoderPtr;

	cv::cudacodec::FormatInfo videoformat;
	videoformat.codec = cv::cudacodec::JPEG;
	videoformat.chromaFormat = cv::cudacodec::YUV420;
	videoformat.width = 4000;
	videoformat.height = 3008;
	decoderPtr.reset(new cv::cudacodec::detail::VideoDecoder(videoformat, lock));

	CUVIDPARSERDISPINFO displayInfo;
	displayInfo.progressive_frame = 1;
	int num_fields = 1;

	CUVIDPROCPARAMS videoProcParams;
	std::memset(&videoProcParams, 0, sizeof(CUVIDPROCPARAMS));
	videoProcParams.progressive_frame = displayInfo.progressive_frame;
	videoProcParams.second_field = 1;
	videoProcParams.top_field_first = 0;
	videoProcParams.unpaired_field = 0;
	videoProcParams.raw_input_dptr = (unsigned long long)jpeg_data_d;
	videoProcParams.raw_output_pitch = 0;

	cv::cuda::GpuMat decodedFrame = decoderPtr->mapFrame(0, videoProcParams);


	return 0;
}

int main(int argc, char* argv[]) {

	cv::Mat refImg = cv::imread("E:\\Project\\CUDA_JPEG\\data\\local_00_refBlk.jpg");
	cv::Mat localImg = cv::imread("E:\\Project\\CUDA_JPEG\\data\\local_00_relight.jpg");
	cv::cuda::GpuMat refImg_d, localImg_d, outputImg_d;

	refImg_d.upload(refImg);
	localImg_d.upload(localImg);

	// find correspondence color
	int width = refImg.cols;
	int height = refImg.rows;
	outputImg_d.create(height, width, CV_8UC3);
	std::vector<cv::Scalar> input_vals;
	std::vector<cv::Scalar> output_vals;
	int start_x = 100;
	int start_y = 100;
	int halfsize = 50;
	while (start_x - halfsize > 0 && start_x + halfsize < width) {
		start_y = 100;
		while (start_y - halfsize > 0 && start_y + halfsize < height) {
			cv::Rect rect(start_x - halfsize, start_y - halfsize,
				2 * halfsize, 2 * halfsize);
			cv::Scalar refVal = cv::mean(refImg(rect));
			cv::Scalar localVal = cv::mean(localImg(rect));
			input_vals.push_back(localVal);
			output_vals.push_back(refVal);
			start_y += 50;
		}
		start_x += 50;
	}
	// create device data pointer
	int ptsize = input_vals.size();
	int* pValues_h[3];
	int* pLevels_h[3];
	const Npp32s* pValues[3];
	const Npp32s* pLevels[3];
	NppiSize osize;
	osize.width = width;
	osize.height = height;
	for (int i = 0; i < 3; i++) {
		cudaMalloc(&pValues[i], ptsize);
		cudaMalloc(&pLevels[i], ptsize);
		pValues_h[i] = new int[ptsize];
		pLevels_h[i] = new int[ptsize];
	}
	for (size_t i = 0; i < ptsize; i++) {
		pValues_h[0][i] = (int)input_vals[i].val[0];
		pValues_h[1][i] = (int)input_vals[i].val[1];
		pValues_h[2][i] = (int)input_vals[i].val[2];
		pLevels_h[0][i] = (int)output_vals[i].val[0];
		pLevels_h[1][i] = (int)output_vals[i].val[1];
		pLevels_h[2][i] = (int)output_vals[i].val[2];
	}

	std::sort(pValues_h[0], pValues_h[0] + ptsize);
	std::sort(pValues_h[1], pValues_h[1] + ptsize);
	std::sort(pValues_h[2], pValues_h[2] + ptsize);
	std::sort(pLevels_h[0], pLevels_h[0] + ptsize);
	std::sort(pLevels_h[1], pLevels_h[1] + ptsize);
	std::sort(pLevels_h[2], pLevels_h[2] + ptsize);


	cudaMemcpy((void*)pValues[0], pValues_h[0], ptsize, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)pValues[1], pValues_h[1], ptsize, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)pValues[2], pValues_h[2], ptsize, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)pLevels[0], pLevels_h[0], ptsize, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)pLevels[1], pLevels_h[1], ptsize, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)pLevels[2], pLevels_h[2], ptsize, cudaMemcpyHostToDevice);

	int nLevels[3];
	nLevels[0] = 256;
	nLevels[1] = 256;
	nLevels[2] = 256;

	NppStatus status = nppiLUT_Linear_8u_C3R(localImg_d.ptr<Npp8u>(), localImg_d.step,
		outputImg_d.ptr<Npp8u>(), outputImg_d.step, osize, pValues, pLevels, nLevels);

	cv::Mat outImg;
	outputImg_d.download(outImg);

	return 0;
}