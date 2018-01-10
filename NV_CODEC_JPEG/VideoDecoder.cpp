/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "VideoDecoder.h"

#include "FrameQueue.h"
#include "stdio.h"
#include <cstring>
#include <cassert>
#include <string>

VideoDecoder::VideoDecoder(const CUVIDEOFORMAT &rVideoFormat,
                           CUcontext &rContext,
                           cudaVideoCreateFlags eCreateFlags,
                           CUvideoctxlock &vidCtxLock)
    : m_VidCtxLock(vidCtxLock)
{
    // get a copy of the CUDA context
    m_Context          = rContext;
    m_VideoCreateFlags = eCreateFlags;

    printf("> VideoDecoder::cudaVideoCreateFlags = <%d>", (int)eCreateFlags);

    switch (eCreateFlags)
    {
        case cudaVideoCreate_Default:
            printf("Default (VP)\n");
            break;

        case cudaVideoCreate_PreferCUDA:
            printf("Use CUDA decoder\n");
            break;

        case cudaVideoCreate_PreferDXVA:
            printf("Use DXVA decoder\n");
            break;

        case cudaVideoCreate_PreferCUVID:
            printf("Use CUVID decoder\n");
            break;

        default:
            printf("Unknown value\n");
            break;
    }

    printf("\n");

    // Validate video format.  These are the currently supported formats via NVDECODE
    assert(cudaVideoCodec_MPEG1    == rVideoFormat.codec ||
           cudaVideoCodec_MPEG2    == rVideoFormat.codec ||
           cudaVideoCodec_MPEG4    == rVideoFormat.codec ||
           cudaVideoCodec_VC1      == rVideoFormat.codec ||
           cudaVideoCodec_H264     == rVideoFormat.codec ||
           cudaVideoCodec_JPEG     == rVideoFormat.codec ||
           cudaVideoCodec_H264_SVC == rVideoFormat.codec ||
           cudaVideoCodec_H264_MVC == rVideoFormat.codec ||
           cudaVideoCodec_HEVC     == rVideoFormat.codec ||
           cudaVideoCodec_VP8      == rVideoFormat.codec ||
           cudaVideoCodec_VP9      == rVideoFormat.codec ||
           cudaVideoCodec_YUV420   == rVideoFormat.codec ||
           cudaVideoCodec_YV12     == rVideoFormat.codec ||
           cudaVideoCodec_NV12     == rVideoFormat.codec ||
           cudaVideoCodec_YUYV     == rVideoFormat.codec ||
           cudaVideoCodec_UYVY     == rVideoFormat.codec);

    assert(cudaVideoChromaFormat_Monochrome == rVideoFormat.chroma_format ||
           cudaVideoChromaFormat_420        == rVideoFormat.chroma_format ||
           cudaVideoChromaFormat_422        == rVideoFormat.chroma_format ||
           cudaVideoChromaFormat_444        == rVideoFormat.chroma_format);

    // Fill the decoder-create-info struct from the given video-format struct.
    memset(&oVideoDecodeCreateInfo_, 0, sizeof(CUVIDDECODECREATEINFO));
    // Create video decoder
    oVideoDecodeCreateInfo_.CodecType           = rVideoFormat.codec;
    oVideoDecodeCreateInfo_.ulWidth             = rVideoFormat.coded_width;
    oVideoDecodeCreateInfo_.ulHeight            = rVideoFormat.coded_height;
    oVideoDecodeCreateInfo_.ulNumDecodeSurfaces = 8;
    if ((oVideoDecodeCreateInfo_.CodecType == cudaVideoCodec_H264) ||
        (oVideoDecodeCreateInfo_.CodecType == cudaVideoCodec_H264_SVC) ||
        (oVideoDecodeCreateInfo_.CodecType == cudaVideoCodec_H264_MVC))
    {
        // assume worst-case of 20 decode surfaces for H264
        oVideoDecodeCreateInfo_.ulNumDecodeSurfaces = 20;
    }
    if (oVideoDecodeCreateInfo_.CodecType == cudaVideoCodec_VP9)
        oVideoDecodeCreateInfo_.ulNumDecodeSurfaces = 12;
    if (oVideoDecodeCreateInfo_.CodecType == cudaVideoCodec_HEVC)
    {
        // ref HEVC spec: A.4.1 General tier and level limits
        int MaxLumaPS = 35651584; // currently assuming level 6.2, 8Kx4K
        int MaxDpbPicBuf = 6;
        int PicSizeInSamplesY = oVideoDecodeCreateInfo_.ulWidth * oVideoDecodeCreateInfo_.ulHeight;
        int MaxDpbSize;
        if (PicSizeInSamplesY <= (MaxLumaPS>>2))
            MaxDpbSize = MaxDpbPicBuf * 4;
        else if (PicSizeInSamplesY <= (MaxLumaPS>>1))
            MaxDpbSize = MaxDpbPicBuf * 2;
        else if (PicSizeInSamplesY <= ((3*MaxLumaPS)>>2))
            MaxDpbSize = (MaxDpbPicBuf * 4) / 3;
        else
            MaxDpbSize = MaxDpbPicBuf;
        MaxDpbSize = MaxDpbSize < 16 ? MaxDpbSize : 16;
        oVideoDecodeCreateInfo_.ulNumDecodeSurfaces = MaxDpbSize + 4;
    }
    oVideoDecodeCreateInfo_.ChromaFormat        = rVideoFormat.chroma_format;
    oVideoDecodeCreateInfo_.OutputFormat        = rVideoFormat.bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
    oVideoDecodeCreateInfo_.bitDepthMinus8      = rVideoFormat.bit_depth_luma_minus8;
    oVideoDecodeCreateInfo_.DeinterlaceMode     = cudaVideoDeinterlaceMode_Adaptive;

    // No scaling
    oVideoDecodeCreateInfo_.ulTargetWidth       = rVideoFormat.display_area.right - rVideoFormat.display_area.left;
    oVideoDecodeCreateInfo_.ulTargetHeight      = rVideoFormat.display_area.bottom - rVideoFormat.display_area.top;
    oVideoDecodeCreateInfo_.display_area.left   = 0;
    oVideoDecodeCreateInfo_.display_area.right  = (short)oVideoDecodeCreateInfo_.ulTargetWidth;
    oVideoDecodeCreateInfo_.display_area.top    = 0;
    oVideoDecodeCreateInfo_.display_area.bottom = (short)oVideoDecodeCreateInfo_.ulTargetHeight;

    oVideoDecodeCreateInfo_.ulNumOutputSurfaces = MAX_FRAME_COUNT;  // We won't simultaneously map more than 8 surfaces
    oVideoDecodeCreateInfo_.ulCreationFlags     = m_VideoCreateFlags;
    oVideoDecodeCreateInfo_.vidLock             = vidCtxLock;
    // create the decoder
    CUresult oResult = cuvidCreateDecoder(&oDecoder_, &oVideoDecodeCreateInfo_);
    if (CUDA_SUCCESS != oResult) {
        printf("cuvidCreateDecoder() failed (error=%d). The combination of parameters isn't supported on the given GPU.\n", oResult);
        exit(1);
    }
}

VideoDecoder::~VideoDecoder()
{
    cuvidDestroyDecoder(oDecoder_);
}

cudaVideoCodec
VideoDecoder::codec()
const
{
    return oVideoDecodeCreateInfo_.CodecType;
}

cudaVideoChromaFormat
VideoDecoder::chromaFormat()
const
{
    return oVideoDecodeCreateInfo_.ChromaFormat;
}

unsigned long
VideoDecoder::maxDecodeSurfaces()
const
{
    return oVideoDecodeCreateInfo_.ulNumDecodeSurfaces;
}

unsigned long
VideoDecoder::frameWidth()
const
{
    return oVideoDecodeCreateInfo_.ulWidth;
}

unsigned long
VideoDecoder::frameHeight()
const
{
    return oVideoDecodeCreateInfo_.ulHeight;
}

unsigned long
VideoDecoder::targetWidth()
const
{
    return oVideoDecodeCreateInfo_.ulTargetWidth;
}

unsigned long
VideoDecoder::targetHeight()
const
{
    return oVideoDecodeCreateInfo_.ulTargetHeight;
}

CUresult
VideoDecoder::decodePicture(CUVIDPICPARAMS *pPictureParameters, CUcontext *pContext)
{
    // Handle CUDA picture decode (this actually calls the hardware VP/CUDA to decode video frames)
    CUresult oResult = cuvidDecodePicture(oDecoder_, pPictureParameters);
    return oResult;
}

CUresult
VideoDecoder::mapFrame(int iPictureIndex, CUdeviceptr *ppDevice, unsigned int *pPitch, CUVIDPROCPARAMS *pVideoProcessingParameters)
{
    CUresult oResult = cuvidMapVideoFrame(oDecoder_,
                                          iPictureIndex,
                                          ppDevice,
                                          pPitch, pVideoProcessingParameters);
    if (ppDevice == NULL)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (*pPitch == 0)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return oResult;
}

CUresult
VideoDecoder::unmapFrame(CUdeviceptr pDevice)
{
    CUresult oResult = cuvidUnmapVideoFrame(oDecoder_, pDevice);
    return oResult;
}

