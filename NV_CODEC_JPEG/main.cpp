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

/* This example demonstrates how to use the Video Decode Library with CUDA
 * bindings to interop between NVDECODE(using CUDA surfaces) and DX9 textures.  
 * Post-Process video (de-interlacing) is suported with this sample.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#endif

// CUDA Header includes
#include "dynlink_nvcuvid.h"  // <nvcuvid.h>
#include "dynlink_cuda.h"     // <cuda.h>
#include "dynlink_cudaD3D11.h" // <cudaD3D11.h>
#include "dynlink_builtin_types.h"    // <builtin_types.h>

// CUDA utilities and system includes
#include "helper_functions.h"
#include "helper_cuda_drvapi.h"

// cudaDecodeD3D11 related helper functions
#include "FrameQueue.h"
#include "VideoSource.h"
#include "VideoParser.h"
#include "VideoDecoder.h"
#include "ImageDX.h"

#include "cudaProcessFrame.h"
#include "cudaModuleMgr.h"

// Include files
#include <math.h>
#include <memory>
#include <iostream>
#include <cassert>

const char *sAppName     = "NVDECODE/D3D11 Video Decoder";
const char *sAppFilename = "NVDecodeD3D11";
const char *sSDKname     = "NVDecodeD3D11";

#define VIDEO_SOURCE_FILE "plush1_720p_10s.m2v"

#ifdef _DEBUG
#define ENABLE_DEBUG_OUT    0
#else
#define ENABLE_DEBUG_OUT    0
#endif

StopWatchInterface *frame_timer  = NULL;
StopWatchInterface *global_timer = NULL;

int                 g_DeviceID    = 0;
bool                g_bWindowed   = true;
bool                g_bDone       = false;
bool                g_bRunning    = false;
bool                g_bAutoQuit   = false;
bool                g_bUseVsync   = false;
bool                g_bFrameRepeat= false;
bool                g_bFrameStep  = false;
bool                g_bQAReadback = false;
bool                g_bFirstFrame = true;
bool                g_bLoop       = false;
bool                g_bUpdateCSC  = true;
bool                g_bUpdateAll  = false;
bool                g_bUseDisplay = true; // this flag enables/disables video on the window
bool                g_bUseInterop = true;
bool                g_bReadback   = false; // this flag enables/disables reading back of a video from a window
bool                g_bWriteFile  = false; // this flag enables/disables writing of a file
bool                g_bPSNR       = false; // if this flag is set true, then we want to compute the PSNR
bool                g_bIsProgressive = true; // assume it is progressive, unless otherwise noted
bool                g_bException  = false;
bool                g_bWaived     = false;
int                 g_nBitDepth   = 0;
int                 g_iRepeatFactor = 1; // 1:1 assumes no frame repeats
long                g_nFrameStart = -1;
long                g_nFrameEnd = -1;
long long           g_nmse_luma = 0;
long long           g_nmse_luma_count = 0;
long long           g_nmse_chroma = 0;
long long           g_nmse_chroma_count = 0;

HWND                g_hWnd = NULL;
WNDCLASSEX          *g_wc = NULL;

int   *pArgc = NULL;
char **pArgv = NULL;

FILE *fpWriteYUV = NULL;
FILE *fpRefYUV = NULL;

cudaVideoCreateFlags g_eVideoCreateFlags = cudaVideoCreate_PreferCUVID;
CUvideoctxlock       g_CtxLock = NULL;

float present_fps, decoded_fps, total_time = 0.0f;

ID3D11Device  *g_pD3DDevice;
ID3D11DeviceContext *g_pContext;
IDXGISwapChain *g_pSwapChain;

// These are CUDA function pointers to the CUDA kernels
CUmoduleManager   *g_pCudaModule;

CUmodule           cuModNV12toARGB       = 0;
CUfunction         g_kernelNV12toARGB    = 0;
CUfunction         g_kernelPassThru      = 0;

CUcontext          g_oContext = 0;
CUdevice           g_oDevice  = 0;

CUstream           g_ReadbackSID = 0, g_KernelSID = 0;

eColorSpace        g_eColorSpace = ITU601;
float              g_nHue        = 0.0f;

// System Memory surface we want to readback to
BYTE          *g_pFrameYUV[6] = { 0, 0, 0, 0, 0, 0 };
FrameQueue    *g_pFrameQueue   = 0;
VideoSource   *g_pVideoSource  = 0;
VideoParser   *g_pVideoParser  = 0;
VideoDecoder  *g_pVideoDecoder = 0;

ImageDX       *g_pImageDX      = 0;
CUdeviceptr    g_pInteropFrame[3] = { 0, 0, 0 }; // if we're using CUDA malloc
CUdeviceptr    g_pRgba = 0;
CUarray        g_backBufferArray = 0;

CUVIDEOFORMAT g_stFormat;

std::string sFileName;

char exec_path[256];

unsigned int g_nWindowWidth  = 0;
unsigned int g_nWindowHeight = 0;

unsigned int g_nVideoWidth  = 0;
unsigned int g_nVideoHeight = 0;

unsigned int g_FrameCount = 0;
unsigned int g_DecodeFrameCount = 0;
unsigned int g_fpsCount = 0;      // FPS count for averaging
unsigned int g_fpsLimit = 16;     // FPS limit for sampling timer;

// Forward declarations
bool    initD3D11(HWND hWnd, int argc, char **argv, int *pbTCC);
HRESULT initD3D11Surface(unsigned int nWidth, unsigned int nHeight);
HRESULT freeDestSurface();
void shutdown();

bool loadVideoSource(const char *video_file,
                     unsigned int &width, unsigned int &height,
                     unsigned int &dispWidth, unsigned int &dispHeight);
void initCudaVideo();

void freeCudaResources(bool bDestroyContext);

bool copyDecodedFrameToTexture(unsigned int &nRepeats, int bUseInterop, int *pbIsProgressive);
void cudaPostProcessFrame(CUdeviceptr *ppDecodedFrame, size_t nDecodedPitch, int nBytesPerSample,
                          CUdeviceptr *ppTextureData,  size_t nTexturePitch,
                          CUmodule cuModNV12toARGB,
                          CUfunction fpCudaKernel, CUstream streamID);
void cudaPostProcessFrame(CUdeviceptr *ppDecodedFrame, size_t nDecodedPitch, int nBytesPerSample,
                          CUarray array,
                          CUmodule cuModNV12toARGB,
                          CUfunction fpCudaKernel, CUstream streamID);
HRESULT drawScene(int field_num);
HRESULT cleanup(bool bDestroyContext);
HRESULT initCudaResources(int argc, char **argv, int bUseInterop, int bTCC);

void renderVideoFrame(HWND hWnd, bool bUseInterop);

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#ifndef STRCASECMP
#define STRCASECMP  _stricmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP _strnicmp
#endif
#else // Linux
#ifndef STRCASECMP
#define STRCASECMP  strcasecmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
#endif
#endif


LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

bool checkHW(char *name, char *gpuType, int dev)
{
    char deviceName[256];
    checkCudaErrors(cuDeviceGetName(deviceName, 256, dev));

    STRCPY(name, strlen(deviceName), deviceName);

    if (!strnicmp(deviceName, gpuType, strlen(gpuType)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

void printStatistics()
{
    int   hh, mm, ss, msec;

    present_fps = 1.f / (total_time / (g_FrameCount * 1000.f));
    decoded_fps = 1.f / (total_time / (g_DecodeFrameCount * 1000.f));

    msec = ((int)total_time % 1000);
    ss   = (int)(total_time/1000) % 60;
    mm   = (int)(total_time/(1000*60)) % 60;
    hh   = (int)(total_time/(1000*60*60)) % 60;

    printf("\n[%s] statistics\n", sSDKname);
    printf("\t Video Length (hh:mm:ss.msec)   = %02d:%02d:%02d.%03d\n", hh, mm, ss, msec);

    printf("\t Frames Presented (inc repeats) = %d\n", g_FrameCount);
    printf("\t Average Present Rate     (fps) = %4.2f\n", present_fps);

    printf("\t Frames Decoded   (hardware)    = %d\n", g_DecodeFrameCount);
    printf("\t Average Rate of Decoding (fps) = %4.2f\n", decoded_fps);
}

void computeFPS(HWND hWnd, bool bUseInterop)
{
    sdkStopTimer(&frame_timer);

    if (g_bRunning)
    {
        g_fpsCount++;

        if (!(g_pFrameQueue->isEndOfDecode() && g_pFrameQueue->isEmpty()))
        {
            g_FrameCount++;
        }
    }

    char sFPS[256];
    std::string sDecodeStatus;

    if (g_pFrameQueue->isEndOfDecode() && g_pFrameQueue->isEmpty())
    {
        sDecodeStatus = "STOP (End of File)\0";

        // we only want to record this once
        if (total_time == 0.0f)
        {
            total_time = sdkGetTimerValue(&global_timer);
        }

        sdkStopTimer(&global_timer);

        if (g_bAutoQuit)
        {
            g_bRunning = false;
            g_bDone    = true;
        }
    }
    else
    {
        if (!g_bRunning)
        {
            sDecodeStatus = "PAUSE\0";
            sprintf(sFPS, "%s [%s] - [%s %d] - Video Display %s / Vsync %s",
                    sAppName, sDecodeStatus.c_str(),
                    (g_bIsProgressive ? "Frame" : "Field"), g_DecodeFrameCount,
                    g_bUseDisplay ? "ON" : "OFF",
                    g_bUseVsync   ? "ON" : "OFF");

            if (bUseInterop && (!g_bQAReadback))
            {
                SetWindowText(hWnd, sFPS);
                UpdateWindow(hWnd);
            }
        }
        else
        {
            if (g_bFrameStep)
            {
                sDecodeStatus = "STEP\0";
            }
            else
            {
                sDecodeStatus = "PLAY\0";
            }
        }

        if (g_fpsCount == g_fpsLimit)
        {
            float ifps = 1.f / (sdkGetAverageTimerValue(&frame_timer) / 1000.f);

            sprintf(sFPS, "[%s] [%s] - [%3.1f fps, %s %d] - Video Display %s / Vsync %s",
                    sAppName, sDecodeStatus.c_str(), ifps,
                    (g_bIsProgressive ? "Frame" : "Field"), g_DecodeFrameCount,
                    g_bUseDisplay ? "ON" : "OFF",
                    g_bUseVsync   ? "ON" : "OFF");

            if (bUseInterop && (!g_bQAReadback))
            {
                SetWindowText(hWnd, sFPS);
                UpdateWindow(hWnd);
            }

            printf("[%s] - [%s: %04d, %04.1f fps, time: %04.2f (ms) ]\n",
                   sSDKname, (g_bIsProgressive ? "Frame" : "Field"), g_FrameCount, ifps, 1000.f/ifps);

            sdkResetTimer(&frame_timer);
            g_fpsCount = 0;
        }
    }

    sdkStartTimer(&frame_timer);
}

HRESULT initCudaResources(int argc, char **argv, int bUseInterop, int bTCC)
{
    HRESULT hr = S_OK;

    CUdevice cuda_device;

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        cuda_device = getCmdLineArgumentInt(argc, (const char **) argv, "device");
        cuda_device = findCudaDeviceDRV(argc, (const char **)argv);

        if (cuda_device < 0)
        {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }

        checkCudaErrors(cuDeviceGet(&g_oDevice, cuda_device));
    }
    else
    {
        cuda_device = gpuGetMaxGflopsDeviceIdDRV();
        checkCudaErrors(cuDeviceGet(&g_oDevice, cuda_device));
    }

    // get compute capabilities and the devicename
    int major, minor;
    size_t totalGlobalMem;
    char deviceName[256];
    checkCudaErrors(cuDeviceComputeCapability(&major, &minor, g_oDevice));
    checkCudaErrors(cuDeviceGetName(deviceName, 256, g_oDevice));
    printf("> Using GPU Device %d: %s has SM %d.%d compute capability\n", cuda_device, deviceName, major, minor);

    checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, g_oDevice));
    printf("  Total amount of global memory:     %4.4f MB\n", (float)totalGlobalMem/(1024*1024));

    // Create CUDA Device w/ D3D11 interop (if WDDM), otherwise CUDA w/o interop (if TCC)
    // (use CU_CTX_BLOCKING_SYNC for better CPU synchronization)
    if (bUseInterop)
    {
        checkCudaErrors(cuD3D11CtxCreate(&g_oContext, &g_oDevice, CU_CTX_BLOCKING_SYNC, g_pD3DDevice));
    }
    else
    {
        checkCudaErrors(cuCtxCreate(&g_oContext, CU_CTX_BLOCKING_SYNC, g_oDevice));
    }

    try
    {
        // Initialize CUDA releated Driver API (32-bit or 64-bit), depending the platform running
        if (sizeof(void *) == 4)
        {
            g_pCudaModule = new CUmoduleManager("NV12ToARGB_drvapi_Win32.ptx", exec_path, 2, 2, 2);
        }
        else
        {
            g_pCudaModule = new CUmoduleManager("NV12ToARGB_drvapi_x64.ptx", exec_path, 2, 2, 2);
        }
    }
    catch (char const *p_file)
    {
        // If the CUmoduleManager constructor fails to load the PTX file, it will throw an exception
        printf("\n>> CUmoduleManager::Exception!  %s not found!\n", p_file);
        printf(">> Please rebuild NV12ToARGB_drvapi.cu or re-install this sample.\n");
        return E_FAIL;
    }

    g_pCudaModule->GetCudaFunction("NV12ToARGB_drvapi", &g_kernelNV12toARGB);
    g_pCudaModule->GetCudaFunction("Passthru_drvapi",   &g_kernelPassThru);

    /////////////////Change///////////////////////////
    // Now we create the CUDA resources and the CUDA decoder context
    initCudaVideo();

    if (bUseInterop)
    {
        initD3D11Surface(g_pVideoDecoder->targetWidth(),
                        g_pVideoDecoder->targetHeight());
        checkCudaErrors(cuMemAlloc(&g_pRgba, g_pVideoDecoder->targetWidth() * g_pVideoDecoder->targetHeight() * 4));
    }
    else
    {
        checkCudaErrors(cuMemAlloc(&g_pInteropFrame[0], g_pVideoDecoder->targetWidth() * g_pVideoDecoder->targetHeight() * 4));
        checkCudaErrors(cuMemAlloc(&g_pInteropFrame[1], g_pVideoDecoder->targetWidth() * g_pVideoDecoder->targetHeight() * 4));
    }

    CUcontext cuCurrent = NULL;
    CUresult result = cuCtxPopCurrent(&cuCurrent);

    if (result != CUDA_SUCCESS)
    {
        printf("cuCtxPopCurrent: %d\n", result);
        assert(0);
    }

    /////////////////////////////////////////
    return ((g_pCudaModule && g_pVideoDecoder && (g_pImageDX || g_pInteropFrame[0])) ? S_OK : E_FAIL);
}

HRESULT reinitCudaResources()
{
    // Free resources
    cleanup(false);

    // Reinit VideoSource and Frame Queue
    g_bIsProgressive = loadVideoSource(sFileName.c_str(),
                                       g_nVideoWidth, g_nVideoHeight,
                                       g_nWindowWidth, g_nWindowHeight);

    /////////////////Change///////////////////////////
    initCudaVideo();
    initD3D11Surface(g_pVideoDecoder->targetWidth(),
                    g_pVideoDecoder->targetHeight());
    /////////////////////////////////////////

    return S_OK;
}

void displayHelp()
{
    printf("\n");
    printf("%s - Help\n\n", sAppName);
    printf("  %s [parameters] -i=source.264 -o=output.yuv\n\n", sAppFilename);
    printf("Program parameters:\n");
    printf("\t-i=source.264   - input file for decoding\n");
    printf("\t-o=output.yuv   - specify base Input file for YUV output\n");
    printf("\t-psnr=ref.yuv   - compare PSNR against reference YUV\n");
    printf("\t-pass=<threshold> - PSNR threshold for PASS/FAIL test\n");
    printf("\t-decodecuda     - Use CUDA kernels for MPEG-2 (Available with 64+ CUDA cores)\n");
    printf("\t-decodedxva     - Use NVDEC for MPEG-2, VC-1, H.264, or H.265 decode\n");
    printf("\t-decodecuvid    - Use NVDEC for MPEG-2, VC-1, H.264, or H.265 decode\n");
    printf("\t-vsync          - Enable vertical sync (disabled by default).\n");
    printf("\t-repeatframe    - Enable automatic framerate repeating.\n");
    printf("\t-repeatfactor=n - Force repeat every frame n times.\n");
    printf("\t-updateall      - always update CSC matrices.\n");
    printf("\t-displayvideo   - display video frames on the window\n");
    printf("\t-nodisplay      - do not open a window for display\n");
    printf("\t-nointerop      - create the CUDA context w/o using graphics interop\n");
    printf("\t-readback       - enable readback of frames to system memory\n");
    printf("\t-device=n       - choose a specific GPU device to decode video with\n");
    printf("\t-nframestart=n  - set the start frame number\n");
    printf("\t-nframeend=n    - set the end frame number\n");
    printf("\t-bitdepth=n     - specify bit-depth in case of VP9 clips (default = 8)\n");
}

void parseCommandLineArguments(int argc, char *argv[])
{
    char video_file[256], yuv_file[256], ref_yuv[256];
    bool bUseDefaultInputFile = true;

    printf("Command Line Arguments:\n");

    for (int n = 0; n < argc; n++)
    {
        printf("argv[%d] = %s\n", n, argv[n]);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        displayHelp();
        exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "i"))
    {
        char *temp;
        getCmdLineArgumentString(argc, (const char **)argv, "i", &temp);
        strcpy(video_file, temp);
        bUseDefaultInputFile = false;
    }

    // Search all command file parameters for video files with extensions:
    // mp4, avc, mkv, 264, h264. vc1, wmv, mp2, mpeg2, mpg
    if (checkCmdLineFlag(argc, (const char **)argv, "o"))
    {
        char *temp;
        getCmdLineArgumentString(argc, (const char **)argv, "o", &temp);
        strcpy(yuv_file, temp);
        g_bReadback = true;
        g_bWriteFile = true;
    }

    // Search all command file parameters for video files with extensions:
    // mp4, avc, mkv, 264, h264. vc1, wmv, mp2, mpeg2, mpg
    if (checkCmdLineFlag(argc, (const char **)argv, "psnr"))
    {
        char *temp;
        getCmdLineArgumentString(argc, (const char **)argv, "psnr", &temp);
        strcpy(ref_yuv, temp);
        g_bReadback = true;
        g_bPSNR = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "decodecuda"))
    {
        g_eVideoCreateFlags = cudaVideoCreate_PreferCUDA;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "decodedxva"))
    {
        g_eVideoCreateFlags = cudaVideoCreate_PreferDXVA;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "decodecuvid"))
    {
        g_eVideoCreateFlags = cudaVideoCreate_PreferCUVID;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "vsync"))
    {
        g_bUseVsync = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "repeatframe"))
    {
        g_bFrameRepeat = true;
        printf("> Framerate Repeating Enabled\n");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "repeatfactor"))
    {
        g_iRepeatFactor = getCmdLineArgumentInt(argc, (const char **)argv, "repeatfactor");
        printf("g_iRepeatFactor = %d\n", g_iRepeatFactor);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "framestep"))
    {
        g_bFrameStep = true;
        g_bUseDisplay = true;
        g_bUseInterop = true;
        g_fpsLimit = 1;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "updateall"))
    {
        g_bUpdateAll = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "displayvideo"))
    {
        g_bUseDisplay = true;
        g_bUseInterop = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "nointerop"))
    {
        g_bUseInterop = false;
        printf("NVDECODE/DirectX graphics interop disabled\n");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "nodisplay"))
    {
        g_bUseDisplay = false;
        printf("Video Window Display disabled\n");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "readback"))
    {
        g_bReadback = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        g_DeviceID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        g_bUseDisplay = true;
        g_bUseInterop = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "bitdepth"))
    {
        g_nBitDepth = getCmdLineArgumentInt(argc, (const char **)argv, "bitdepth");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "nframestart"))
    {
        g_nFrameStart = getCmdLineArgumentInt(argc, (const char **)argv, "nframestart");
        printf("YUV output @ nStartFrame = %d\n", g_nFrameStart);
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "nframeend"))
    {
        g_nFrameEnd   = getCmdLineArgumentInt(argc, (const char **)argv, "nframeend");
        printf("YUV output @ nStartEnd = %d\n", g_nFrameEnd);
    }

    if (g_bUseDisplay == false)
    {
        g_bQAReadback = true;
        g_bUseInterop = false;
    }

    if (g_bLoop == false)
    {
        g_bAutoQuit = true;
    }

    if (bUseDefaultInputFile)
    {
        strcpy(video_file, sdkFindFilePath(VIDEO_SOURCE_FILE, argv[0]));
    }

    // Now verify the input video file is legit
    FILE *fp = NULL;
    FOPEN(fp, video_file, "r");
    if (video_file == NULL && fp == NULL)
    {
        printf("[%s]: unable to find file: [%s]\nExiting...\n", sAppFilename, VIDEO_SOURCE_FILE);
        exit(EXIT_FAILURE);
    }

    if (fp)
    {
        printf("[%s]: input file:  [%s]\n", sAppFilename, video_file);
        fclose(fp);
    }

    // Now verify the input reference YUV file is legit
    FOPEN(fpRefYUV, ref_yuv, "rb");
    if (ref_yuv == NULL && fpRefYUV == NULL)
    {
        printf("[%s]: unable to find file: [%s]\nExiting...\n", sAppFilename, ref_yuv);
        exit(EXIT_FAILURE);
    }

    // default video file loaded by this sample
    sFileName = video_file;

    if (g_bWriteFile && strlen(yuv_file) > 0)
    {
        printf("[%s]: output file: [%s]\n", sAppFilename, yuv_file);

        FOPEN(fpWriteYUV, yuv_file, "wb");
        if (fpWriteYUV == NULL)
        {
            printf("Error opening file [%s]\n", yuv_file);
        }
    }

    // store the current path so we can reinit the CUDA context
    strcpy(exec_path, argv[0]);
}

static long long SumSquareError(const unsigned char *src1, const unsigned char *src2, unsigned int count)
{
    long long sum = 0;
    for (unsigned int i = 0; i<count; i++)
    {
        int diff = src1[i] - src2[i];
        sum += diff*diff;
    }
    return sum;
}

inline double PSNR(long long sse, long long count)
{
    return 10 * log10(255.0*255.0*(double)count / (double)sse);
}

template <class T>
void SaveFrameAsYUV(T *pdst,
    const T *psrc,
    int width, int height, int pitch)
{
    int w2 = width;
    int h2 = height;
    int luma_size = w2*h2*sizeof(T);
    int chroma_size = w2*(h2 >> 1)*sizeof(T);
    int x, y, width_2, height_2;
    int xy_offset = width*height;
    int uvoffs = (width / 2)*(height / 2);
    const T *py = psrc;
    const T *puv = psrc + height*pitch / sizeof(T);

    if (((long)g_DecodeFrameCount >= g_nFrameStart) &&
        ((long)g_DecodeFrameCount <= g_nFrameEnd)
        )
    {
        //      printf(" Saving YUV Frame %d (start,end)=(%d,%d)\n", g_DecodeFrameCount, g_nFrameStart, g_nFrameEnd);
        printf("%d+", g_DecodeFrameCount);
    }
    else if ((g_nFrameStart == -1) && (g_nFrameEnd == -1))
    {
        printf("+");
    }
    else // we do nothing and exit
    {
        return;
    }

    // luma
    for (y = 0; y<height; y++)
    {
        memcpy(&pdst[y*width], py, width * sizeof(T));
        py += pitch / sizeof(T);
    }

    // De-interleave chroma
    width_2 = width >> 1;
    height_2 = height >> 1;
    for (y = 0; y<height_2; y++)
    {
        for (x = 0; x<width_2; x++)
        {
            pdst[xy_offset + y*(width_2)+x] = puv[x * 2];
            pdst[xy_offset + uvoffs + y*(width_2)+x] = puv[x * 2 + 1];
        }
        puv += pitch / sizeof(T);
    }

    if (g_bWriteFile) // for writing to file
    {
        fwrite(pdst, 1, (width*height + (width*height) / 2) * sizeof(T), fpWriteYUV);
    }
    if (g_bPSNR) // for calculating PSNR
    {
        unsigned char *iyuv_ref = new unsigned char[luma_size + chroma_size + 16]; // pointer to refernce file data
        long long frm_mse_y = 0;
        long long frm_mse_uv = 0;
        fseek(fpRefYUV, (luma_size + chroma_size)*(g_DecodeFrameCount - 1), SEEK_SET);
        if (fread(iyuv_ref, 1, luma_size + chroma_size, fpRefYUV))
        {
            double psnr = 0;
            frm_mse_y = SumSquareError((unsigned char *)pdst, iyuv_ref, luma_size);
            frm_mse_uv = SumSquareError((unsigned char *)pdst + luma_size, iyuv_ref + luma_size, chroma_size);
            g_nmse_luma += frm_mse_y;
            g_nmse_chroma += frm_mse_uv;
            g_nmse_luma_count += luma_size;
            g_nmse_chroma_count += chroma_size;
            psnr = PSNR(frm_mse_y + frm_mse_uv, luma_size + chroma_size);
            if (g_nmse_luma + g_nmse_chroma > 0) // Condition to check whether PSNR for current frame is correct or not
                printf(" Value of PSNR : %lf\n", psnr);
        }
        else
        {
            printf("EOF! reference file read failed\n");
        }
        delete iyuv_ref;
    }

}

int main(int argc, char *argv[])
{
    pArgc = &argc;
    pArgv = argv;

    sdkCreateTimer(&frame_timer);
    sdkResetTimer(&frame_timer);

    sdkCreateTimer(&global_timer);
    sdkResetTimer(&global_timer);

    // parse the command line arguments
    parseCommandLineArguments(argc, argv);

    // create window (after we know the size of the input file size)
    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, MsgProc, 0L, 0L,
                      GetModuleHandle(NULL), NULL, NULL, NULL, NULL,
                      sAppName, NULL
                    };
    RegisterClassEx(&wc);
    g_wc = &wc;

    // figure out the window size we must create to get a *client* area
    // that is of the size requested by m_dimensions.
    RECT adjustedWindowSize;
    DWORD dwWindowStyle;

    // Initialize the CUDA and NVDECODE
    typedef HMODULE CUDADRIVER;
    CUDADRIVER hHandleDriver = 0;
    CUresult cuResult;
    cuResult = cuInit    (0, __CUDA_API_VERSION, hHandleDriver);
    cuResult = cuvidInit (0);

    // Find out the video size
    g_bIsProgressive = loadVideoSource(sFileName.c_str(),
                                       g_nVideoWidth, g_nVideoHeight,
                                       g_nWindowWidth, g_nWindowHeight);

    // Create the Windows
    if (g_bUseDisplay)
    {
        dwWindowStyle = WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS;
        SetRect(&adjustedWindowSize, 0, 0, g_nVideoWidth  , g_nVideoHeight);
        AdjustWindowRect(&adjustedWindowSize, dwWindowStyle, false);

        g_nWindowWidth  = adjustedWindowSize.right  - adjustedWindowSize.left;
        g_nWindowHeight = adjustedWindowSize.bottom - adjustedWindowSize.top;

        // Create the application's window
        g_hWnd = CreateWindow(wc.lpszClassName, sAppName,
                            dwWindowStyle,
                            0, 0,
                            g_nWindowWidth,
                            g_nWindowHeight,
                            NULL, NULL, wc.hInstance, NULL);
    }

    int bTCC = 0;


    if (g_bUseInterop)
    {
        // Initialize Direct3D
        if (initD3D11(g_hWnd, argc, argv, &bTCC) == false)
        {
            g_bAutoQuit = true;
            g_bWaived   = true;
            shutdown();
        }
    }

    if (!g_bUseInterop && g_eVideoCreateFlags == cudaVideoCreate_PreferDXVA)
    {
        // preferDXVA will not work with -nointerop mode. Overwrite it.
        g_eVideoCreateFlags = cudaVideoCreate_PreferCUVID;
    }

    // If we are using TCC driver, then graphics interop must be disabled
    if (bTCC)
    {
        g_bUseInterop = false;
    }

    // Initialize CUDA/D3D11 context and other video memory resources
    if (initCudaResources(argc, argv, g_bUseInterop, bTCC) == E_FAIL)
    {
        g_bAutoQuit  = true;
        g_bException = true;
        g_bWaived    = true;
        shutdown();
    }

    g_pVideoSource->start();
    g_bRunning = true;

    if (!g_bQAReadback && !bTCC)
    {
        ShowWindow(g_hWnd, SW_SHOWDEFAULT);
        UpdateWindow(g_hWnd);
    }

    // the main loop
    sdkStartTimer(&frame_timer);
    sdkStartTimer(&global_timer);
    sdkResetTimer(&global_timer);

    if (!g_bUseInterop)
    {
        // On this case we drive the display with a while loop (no openGL calls)
        while (!g_bDone)
        {
            renderVideoFrame(g_hWnd, g_bUseInterop);
        }
    }
    else
    {
        // Standard windows loop
        while (!g_bDone)
        {
            MSG msg;
            ZeroMemory(&msg, sizeof(msg));

            while (msg.message!=WM_QUIT)
            {
                if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
                {
                    TranslateMessage(&msg);
                    DispatchMessage(&msg);
                }
                else
                {
                    renderVideoFrame(g_hWnd, g_bUseInterop);
                }

                if (g_bAutoQuit && g_bDone)
                {
                    break;
                }
            }
        } // while loop
    }

    // we only want to record this once
    if (total_time == 0.0f)
    {
        total_time = sdkGetTimerValue(&global_timer);
    }
    sdkStopTimer(&global_timer);

    g_pFrameQueue->endDecode();
    g_pVideoSource->stop();

    if (fpWriteYUV != NULL)
    {
        fflush(fpWriteYUV);
        fclose(fpWriteYUV);
        fpWriteYUV = NULL;
    }

    // final output for PSNR
    if (g_bPSNR)
    {
        double psnr = PSNR(g_nmse_luma + g_nmse_chroma, g_nmse_chroma_count + g_nmse_luma_count);
        if (g_nmse_luma + g_nmse_chroma > 0)
            printf("Overall PSNR                 : %6.3fdB (Y=%6.3fdB, UV=%6.3fdB)\n", psnr, PSNR(g_nmse_luma, g_nmse_luma_count), PSNR(g_nmse_chroma, g_nmse_chroma_count));
        else
            printf("Overall PSNR                 : infinite (perfect match)\n");
    }

    printStatistics();

    g_bWaived = false;
    shutdown();
}

void shutdown()
{
    // clean up CUDA and OpenGL resources
    cleanup(g_bWaived ? false : true);

    if (!g_bQAReadback)
    {
        // Unregister windows class
        UnregisterClass(g_wc->lpszClassName, g_wc->hInstance);
    }

    
    if (g_bAutoQuit)
    {
        PostQuitMessage(0);
    }

    if (g_hWnd)
    {
        DestroyWindow(g_hWnd);
    }

    if (g_bWaived)
    {
        exit(EXIT_WAIVED);
    }
    else
    {
        exit(g_bException ? EXIT_FAILURE : EXIT_SUCCESS);
    }
}

inline std::string wcs2mbstring(const wchar_t *wcs)
{
    size_t len = wcslen(wcs) + 1;
    char *mbs = new char[len];
    wcstombs(mbs, wcs, len);

    std::string mbstring(mbs);
    delete mbs;
    return mbstring;
}

// Initialize Direct3D
bool
initD3D11(HWND hWnd, int argc, char **argv, int *pbTCC)
{
    int dev, device_count = 0;
    bool bSpecifyDevice=false;
    char device_name[256];

    // Check for a min spec of Compute 1.1 capability before running
    checkCudaErrors(cuDeviceGetCount(&device_count));

    for (int i=0; i < argc; i++)
    {
        int string_start = 0;

        while (argv[i][string_start++] != '-');

        const char *string_argv = &argv[i][string_start];

        if (!STRNCASECMP(string_argv, "device=", 7))
        {
            bSpecifyDevice = true;
        }
    }

    // If deviceID == 0, and there is more than 1 device, let's find the first available graphics GPU
    if (!bSpecifyDevice && device_count > 0)
    {
        for (int i=0; i < device_count; i++)
        {
            checkCudaErrors(cuDeviceGet(&dev, i));
            checkCudaErrors(cuDeviceGetName(device_name, 256, dev));

            int bSupported = checkCudaCapabilitiesDRV(1, 1, i);

            if (!bSupported)
            {
                printf("  -> GPU: \"%s\" does not meet the minimum spec of SM 1.1\n", device_name);
                printf("  -> A GPU with a minimum compute capability of SM 1.1 or higher is required.\n");
                return false;
            }

            checkCudaErrors(cuDeviceGetAttribute(pbTCC ,  CU_DEVICE_ATTRIBUTE_TCC_DRIVER, dev));
            printf("  -> GPU %d: < %s > driver mode is: %s\n", dev, device_name, *pbTCC ? "TCC" : "WDDM");

            if (*pbTCC)
            {
                g_bUseInterop = false;
                continue;
            }
            else
            {
                g_DeviceID = i; // we choose an available WDDM display device
            }

            printf("\n");
        }
    }
    else
    {
        if ((g_DeviceID > (device_count-1)) || (g_DeviceID < 0))
        {
            printf(" >>> Invalid GPU Device ID=%d specified, only %d GPU device(s) are available.<<<\n", g_DeviceID, device_count);
            printf(" >>> Valid GPU ID (n) range is between [%d,%d]...  Exiting... <<<\n", 0, device_count-1);
            return false;
        }

        // We are specifying a GPU device, check to see if it is TCC or not
        checkCudaErrors(cuDeviceGet(&dev, g_DeviceID));
        checkCudaErrors(cuDeviceGetName(device_name, 256, dev));

        checkCudaErrors(cuDeviceGetAttribute(pbTCC ,  CU_DEVICE_ATTRIBUTE_TCC_DRIVER, dev));
        printf("  -> GPU %d: < %s > driver mode is: %s\n", dev, device_name, *pbTCC ? "TCC" : "WDDM");

        if (*pbTCC)
        {
            g_bUseInterop = false;
        }
    }

    HRESULT eResult = S_OK;

    if (g_bUseInterop)
    {

        bool bDeviceFound = false;
        int device;

        // Find the first CUDA capable device
        CUresult cuStatus;
        IDXGIAdapter *pAdapter = NULL;
        IDXGIFactory1 *pFactory = NULL;
        CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **)&pFactory);
        for (unsigned int g_iAdapter = 0; pFactory->EnumAdapters(g_iAdapter, &pAdapter) == S_OK; g_iAdapter++)
        {
            DXGI_ADAPTER_DESC desc;
            pAdapter->GetDesc(&desc);

            cuStatus = cuD3D11GetDevice(&device, pAdapter);
            printf("> Display Device: \"%s\" %s Direct3D11\n",
                   wcs2mbstring(desc.Description).c_str(),
                   (cuStatus == cudaSuccess) ? "supports" : "does not support");

            if (cudaSuccess == cuStatus)
            {
                if (!bSpecifyDevice) {
                    bDeviceFound = true;
                    break;
                }
                if (g_DeviceID == device) {
                    bDeviceFound = true;
                    break;
                }
            }
        }
        pFactory->Release();

        // we check to make sure we have found a cuda-compatible D3D device to work on
        if (!bDeviceFound)
        {
            printf("\n");
            printf("  No CUDA-compatible Direct3D9 device available\n");
            // destroy the D3D device
            return false;
        }

        // Create the D3D Display Device
        /* Initialize D3D */
        DXGI_SWAP_CHAIN_DESC sc = { 0 };
        sc.BufferCount = 1;
        sc.BufferDesc.Width = g_nVideoWidth;
        sc.BufferDesc.Height = g_nVideoHeight;
        sc.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        sc.BufferDesc.RefreshRate.Numerator = 0;
        sc.BufferDesc.RefreshRate.Denominator = 1;
        sc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sc.OutputWindow = hWnd;
        sc.SampleDesc.Count = 1;
        sc.SampleDesc.Quality = 0;
        sc.Windowed = TRUE;

        HRESULT hr = D3D11CreateDeviceAndSwapChain(pAdapter, D3D_DRIVER_TYPE_UNKNOWN,
            NULL, 0, NULL, 0, D3D11_SDK_VERSION, &sc, &g_pSwapChain, &g_pD3DDevice, NULL, &g_pContext);
        if (FAILED(hr)) {
            printf("Unable to create DX11 device and swapchain, hr=0x%x", hr);
            return false;
        }
    }
    else
    {
        fprintf(stderr, "> %s is decoding w/o visualization\n", sSDKname);
        eResult = S_OK;
    }

    return (eResult == S_OK);
}

// Initialize Direct3D Textures (allocation and initialization)
HRESULT
initD3D11Surface(unsigned int nWidth, unsigned int nHeight)
{
    g_pImageDX = new ImageDX(g_pD3DDevice, g_pContext, g_pSwapChain,
                             nWidth, nHeight,
                             nWidth, nHeight,
                             g_bUseVsync,
                             ImageDX::BGRA_PIXEL_FORMAT); // ImageDX::LUMINANCE_PIXEL_FORMAT
    g_pImageDX->clear(0x80);

    g_pImageDX->setCUDAcontext(g_oContext);
    g_pImageDX->setCUDAdevice(g_oDevice);

    return S_OK;
}

HRESULT
freeDestSurface()
{
    if (g_pImageDX)
    {
        delete g_pImageDX;
        g_pImageDX = NULL;
    }

    return S_OK;
}


bool
loadVideoSource(const char *video_file,
                unsigned int &width    , unsigned int &height,
                unsigned int &dispWidth, unsigned int &dispHeight)
{
    VideoSource *pSource = NULL;
    std::unique_ptr<FrameQueue> apFrameQueue(new FrameQueue);
    try
    {
        pSource = new VideoSource(video_file, apFrameQueue.get(), g_nBitDepth);
    }
    catch (CUresult val)
    {
        printf("VideoSource returned an error (Video Codec is not supported), exiting...\n", val);
        g_bWaived = true;
        shutdown();
    }
    std::unique_ptr<VideoSource> apVideoSource(pSource);

    // retrieve the video source (width,height)
    apVideoSource->getDisplayDimensions(width, height);
    apVideoSource->getDisplayDimensions(dispWidth, dispHeight);

    memset(&g_stFormat, 0, sizeof(CUVIDEOFORMAT));
    std::cout << (g_stFormat = apVideoSource->format()) << std::endl;

    if (g_bFrameRepeat)
    {
        if (apVideoSource->format().frame_rate.denominator > 0)
        {
            g_iRepeatFactor = (int)(60.0f / ceil((float)apVideoSource->format().frame_rate.numerator / (float)apVideoSource->format().frame_rate.denominator));
        }
    }

    printf("Frame Rate Playback Speed = %d fps\n", 60 / g_iRepeatFactor);

    g_pFrameQueue  = apFrameQueue.release();
    g_pVideoSource = apVideoSource.release();

    if (g_pVideoSource->format().codec == cudaVideoCodec_JPEG)
    {
        g_eVideoCreateFlags = cudaVideoCreate_PreferCUDA;
    }

    bool IsProgressive = 0;
    g_pVideoSource->getProgressive(IsProgressive);
    return IsProgressive;
}

void
initCudaVideo()
{
    // bind the context lock to the CUDA context
    CUresult result = cuvidCtxLockCreate(&g_CtxLock, g_oContext);
    CUVIDEOFORMATEX oFormatEx;
    memset(&oFormatEx, 0, sizeof(CUVIDEOFORMATEX));
    oFormatEx.format = g_stFormat;

    if (result != CUDA_SUCCESS)
    {
        printf("cuvidCtxLockCreate failed: %d\n", result);
        assert(0);
    }

    CUVIDEOFORMAT videoFormat = g_pVideoSource->format();
    CUVIDDECODECAPS videoDecodeCaps = {};
    videoDecodeCaps.eCodecType = videoFormat.codec;
    videoDecodeCaps.eChromaFormat = videoFormat.chroma_format;
    videoDecodeCaps.nBitDepthMinus8 = videoFormat.bit_depth_luma_minus8;
    if (cuvidGetDecoderCaps(&videoDecodeCaps) != CUDA_SUCCESS)
    {
        printf("cuvidGetDecoderCaps failed: %d\n", result);
        return;
    }
    if (!videoDecodeCaps.bIsSupported) {
        printf("Error: This video format isn't supported on the selected GPU.");
        exit(1);;
    }
    // If XxY is max supported resolution for a codec on a GPU and YxX resolution
    // is also supported then Max Supported width is Max(X, Y) and Max supported height is Max(X, Y)
    // But Max supported MBCount is XxY / 256.
    // E.g. 4096x2304 is max supported resolution and 2304x4096 is also supported then
    // Max Width = 4096, Max Height = 4096, But Max supported MB Count = 4096*2304 / 256 = 36864
    printf("This video format is supported on the selected GPU. \n"
           "    Min resolution supported : %dx%d\n"
           "    Max width supported      : %d\n"
           "    Max height supported     : %d\n"
           "    Max macroblocks supported: %d\n", videoDecodeCaps.nMinWidth, videoDecodeCaps.nMinHeight, 
           videoDecodeCaps.nMaxWidth, videoDecodeCaps.nMaxHeight, videoDecodeCaps.nMaxMBCount);

    std::unique_ptr<VideoDecoder> apVideoDecoder(new VideoDecoder(g_pVideoSource->format(), g_oContext, g_eVideoCreateFlags, g_CtxLock));
    std::unique_ptr<VideoParser> apVideoParser(new VideoParser(apVideoDecoder.get(), g_pFrameQueue, &oFormatEx));
    g_pVideoSource->setParser(*apVideoParser.get());

    g_pVideoParser  = apVideoParser.release();
    g_pVideoDecoder = apVideoDecoder.release();

    // Create a Stream ID for handling Readback
    if (g_bReadback)
    {
        checkCudaErrors(cuStreamCreate(&g_ReadbackSID, 0));
        checkCudaErrors(cuStreamCreate(&g_KernelSID,   0));
        printf("> initCudaVideo()\n");
        printf("  CUDA Streams (%s) <g_ReadbackSID = %p>\n", ((g_ReadbackSID == 0) ? "Disabled" : "Enabled"), g_ReadbackSID);
        printf("  CUDA Streams (%s) <g_KernelSID   = %p>\n", ((g_KernelSID   == 0) ? "Disabled" : "Enabled"), g_KernelSID);
    }
}


void
freeCudaResources(bool bDestroyContext)
{
    if (g_pVideoParser)
    {
        delete g_pVideoParser;
    }

    if (g_pVideoDecoder)
    {
        delete g_pVideoDecoder;
    }

    if (g_pVideoSource)
    {
        delete g_pVideoSource;
    }

    if (g_pFrameQueue)
    {
        delete g_pFrameQueue;
    }

    if (g_ReadbackSID)
    {
        cuStreamDestroy(g_ReadbackSID);
    }

    if (g_KernelSID)
    {
        cuStreamDestroy(g_KernelSID);
    }

    if (g_CtxLock)
    {
        checkCudaErrors(cuvidCtxLockDestroy(g_CtxLock));
    }

    if (g_oContext && bDestroyContext)
    {
        checkCudaErrors(cuCtxDestroy(g_oContext));
        g_oContext = NULL;
    }     
}

// Run the Cuda part of the computation (if g_pFrameQueue is empty, then return false)
bool copyDecodedFrameToTexture(unsigned int &nRepeats, int bUseInterop, int *pbIsProgressive)
{
    CUVIDPARSERDISPINFO oDisplayInfo;

    if (g_pFrameQueue->dequeue(&oDisplayInfo))
    {
        CCtxAutoLock lck(g_CtxLock);
        // Push the current CUDA context (only if we are using CUDA decoding path)
        CUresult result = cuCtxPushCurrent(g_oContext);

        CUdeviceptr  pDecodedFrame[3] = { 0, 0, 0 };
        CUdeviceptr  pInteropFrame[3] = { 0, 0, 0 };

        *pbIsProgressive = oDisplayInfo.progressive_frame;
        g_bIsProgressive = oDisplayInfo.progressive_frame ? true : false;

        int num_fields = 1;
        if (g_bUseVsync) {            
            num_fields = std::min(2 + oDisplayInfo.repeat_first_field, 3);            
        }
        nRepeats = num_fields;

        CUVIDPROCPARAMS oVideoProcessingParameters;
        memset(&oVideoProcessingParameters, 0, sizeof(CUVIDPROCPARAMS));

        oVideoProcessingParameters.progressive_frame = oDisplayInfo.progressive_frame;
        oVideoProcessingParameters.top_field_first = oDisplayInfo.top_field_first;
        oVideoProcessingParameters.unpaired_field = (oDisplayInfo.repeat_first_field < 0);

        for (int active_field = 0; active_field < num_fields; active_field++)
        {
            unsigned int nDecodedPitch = 0;
            unsigned int nWidth = 0;
            unsigned int nHeight = 0;

            oVideoProcessingParameters.second_field = active_field;

            // map decoded video frame to CUDA surfae
            g_pVideoDecoder->mapFrame(oDisplayInfo.picture_index, &pDecodedFrame[active_field], &nDecodedPitch, &oVideoProcessingParameters);
            nWidth  = g_pVideoDecoder->targetWidth();
            nHeight = g_pVideoDecoder->targetHeight();
            // map DirectX texture to CUDA surface
            size_t nTexturePitch = 0;

            // If we are Encoding and this is the 1st Frame, we make sure we allocate system memory for readbacks
            if (g_bReadback && g_bFirstFrame && g_ReadbackSID)
            {
                CUresult result;
                checkCudaErrors(result = cuMemAllocHost((void **)&g_pFrameYUV[0], (nDecodedPitch * nHeight + nDecodedPitch*nHeight/2)));
                checkCudaErrors(result = cuMemAllocHost((void **)&g_pFrameYUV[1], (nDecodedPitch * nHeight + nDecodedPitch*nHeight/2)));
                checkCudaErrors(result = cuMemAllocHost((void **)&g_pFrameYUV[2], (nDecodedPitch * nHeight + nDecodedPitch*nHeight/2)));
                checkCudaErrors(result = cuMemAllocHost((void **)&g_pFrameYUV[3], (nDecodedPitch * nHeight + nDecodedPitch*nHeight/2)));
                checkCudaErrors(result = cuMemAllocHost((void **)&g_pFrameYUV[4], (nDecodedPitch * nHeight + nDecodedPitch*nHeight / 2)));
                checkCudaErrors(result = cuMemAllocHost((void **)&g_pFrameYUV[5], (nDecodedPitch * nHeight + nDecodedPitch*nHeight / 2)));

                g_bFirstFrame = false;

                if (result != CUDA_SUCCESS)
                {
                    printf("cuMemAllocHost returned %d\n", (int)result);
                    checkCudaErrors(result);
                }
            }

            // If streams are enabled, we can perform the readback to the host while the kernel is executing
            if (g_bReadback && g_ReadbackSID)
            {
                CUresult result = cuMemcpyDtoHAsync(g_pFrameYUV[active_field], pDecodedFrame[active_field], (nDecodedPitch * nHeight * 3 / 2), g_ReadbackSID);

                if (result != CUDA_SUCCESS)
                {
                    printf("cuMemAllocHost returned %d\n", (int)result);
                    checkCudaErrors(result);
                }
            }

#if ENABLE_DEBUG_OUT
            printf("%s = %02d, PicIndex = %02d, OutputPTS = %08d\n",
                   (oDisplayInfo.progressive_frame ? "Frame" : "Field"),
                   g_DecodeFrameCount, oDisplayInfo.picture_index, oDisplayInfo.timestamp);
#endif

            if (g_pImageDX)
            {
                // map the texture surface
                g_pImageDX->map(&g_backBufferArray, active_field);
                cudaPostProcessFrame(&pDecodedFrame[active_field], nDecodedPitch, g_pVideoDecoder->GetNumBytesPerSample(), g_backBufferArray, g_pCudaModule->getModule(), g_kernelNV12toARGB, g_KernelSID);
                // unmap the texture surface
                g_pImageDX->unmap(active_field);
            }
            else
            {
                pInteropFrame[active_field] = g_pInteropFrame[active_field];
                nTexturePitch = g_pVideoDecoder->targetWidth() * 2;
                cudaPostProcessFrame(&pDecodedFrame[active_field], nDecodedPitch, g_pVideoDecoder->GetNumBytesPerSample(), &pInteropFrame[active_field], 
                                     nTexturePitch, g_pCudaModule->getModule(), g_kernelNV12toARGB, g_KernelSID);
            }

            // unmap video frame
            // unmapFrame() synchronizes with the VideoDecode API (ensures the frame has finished decoding)
            g_pVideoDecoder->unmapFrame(pDecodedFrame[active_field]);                  
            g_DecodeFrameCount++;

/*          if (g_bPSNR)
            {
                int w2 = nWidth;
                int h2 = nHeight;
                int luma_size = w2*h2;
                int chroma_size = w2*(h2 >> 1);

                unsigned char *iyuv = new unsigned char[w2*(h2 + (h2 >> 1)) + 16];

//                NV12toIYUV(pRawNV12, pRawNV12 + state.dci.ulTargetHeight*pitch, iyuv, w2, h2, pitch);

                unsigned char *iyuv_ref = new unsigned char[w2*(h2 + (h2 >> 1)) + 16];
                long long frm_mse_y = 0, frm_mse_uv = 0;

                if (fread(iyuv_ref, w2, h2 + (h2 >> 1), fpRefYUV) > 0)
                {
                    double psnr;

                    frm_mse_y = SumSquareError(iyuv, iyuv_ref, luma_size);
                    frm_mse_uv = SumSquareError(iyuv + luma_size, iyuv_ref + luma_size, chroma_size);
                    mse_luma += frm_mse_y;
                    mse_chroma += frm_mse_uv;
                    mse_luma_count += luma_size;
                    mse_chroma_count += chroma_size;
                    psnr = PSNR(frm_mse_y + frm_mse_uv, luma_size + chroma_size);
                    if ((psnr_threshold) && (psnr < (double)psnr_threshold))
                    {
                        printf("  [%d(%d)] PSNR=%6.3fdB (Y=%6.3fdB, UV=%6.3fdB) [size=%d]\n", pic_cnt,
                            state.frmStats[PicIdx].pic_num_in_decode_order,
                            psnr, PSNR(frm_mse_y, luma_size), PSNR(frm_mse_uv, chroma_size),
                            state.frmStats[PicIdx].bytes_in_picture);
                        fflush(stdout);
                    }
                }
                delete iyuv_ref;
            }
            delete iyuv;
*/

            if (g_bWriteFile || g_bPSNR)
            {
                cuStreamSynchronize(g_ReadbackSID);
                if (g_pVideoDecoder->GetNumBytesPerSample() == 1) {
                    SaveFrameAsYUV(g_pFrameYUV[active_field + 3],
                        g_pFrameYUV[active_field],
                        nWidth, nHeight, nDecodedPitch);
                } else {
                    SaveFrameAsYUV((unsigned short *)g_pFrameYUV[active_field + 3],
                        (unsigned short *)g_pFrameYUV[active_field],
                        nWidth, nHeight, nDecodedPitch);
                }            
            }
        }

        // Detach from the Current thread
        checkCudaErrors(cuCtxPopCurrent(NULL));
        // release the frame, so it can be re-used in decoder
        g_pFrameQueue->releaseFrame(&oDisplayInfo);
    }
    else
    {
        // Frame Queue has no frames, we don't compute FPS until we start
        return false;
    }

    // check if decoding has come to an end.
    // if yes, signal the app to shut down.
    if (!g_pVideoSource->isStarted() && g_pFrameQueue->isEndOfDecode() && g_pFrameQueue->isEmpty())
    {
        // Let's free the Frame Data
        if (g_ReadbackSID && g_pFrameYUV)
        {
            cuMemFreeHost((void *)g_pFrameYUV[0]);
            cuMemFreeHost((void *)g_pFrameYUV[1]);
            cuMemFreeHost((void *)g_pFrameYUV[2]);
            cuMemFreeHost((void *)g_pFrameYUV[3]);
            cuMemFreeHost((void *)g_pFrameYUV[4]);
            cuMemFreeHost((void *)g_pFrameYUV[5]);
            g_pFrameYUV[0] = NULL;
            g_pFrameYUV[1] = NULL;
            g_pFrameYUV[2] = NULL;
            g_pFrameYUV[3] = NULL;
            g_pFrameYUV[4] = NULL;
            g_pFrameYUV[5] = NULL;
        }

        // Let's just stop, and allow the user to quit, so they can at least see the results
        g_pVideoSource->stop();

        // If we want to loop reload the video file and restart
        if (g_bLoop && !g_bAutoQuit)
        {
            reinitCudaResources();
            g_FrameCount = 0;
            g_DecodeFrameCount = 0;
            g_pVideoSource->start();
        }

        if (g_bAutoQuit)
        {
            g_bDone = true;
        }
    }

    return true;
}

// This is the CUDA stage for Video Post Processing.  Last stage takes care of the NV12 to ARGB
void
cudaPostProcessFrame(CUdeviceptr *ppDecodedFrame, size_t nDecodedPitch, int nBytesPerSample,
                     CUdeviceptr *ppTextureData,  size_t nTexturePitch,
                     CUmodule cuModNV12toARGB,
                     CUfunction fpCudaKernel, CUstream streamID)
{
    uint32 nWidth  = g_pVideoDecoder->targetWidth();
    uint32 nHeight = g_pVideoDecoder->targetHeight();

    // Upload the Color Space Conversion Matrices
    if (g_bUpdateCSC)
    {
        // CCIR 601/709
        float hueColorSpaceMat[9];
        setColorSpaceMatrix(g_eColorSpace,    hueColorSpaceMat, g_nHue);
        updateConstantMemory_drvapi(cuModNV12toARGB, hueColorSpaceMat);

        if (!g_bUpdateAll)
        {
            g_bUpdateCSC = false;
        }
    }

    // TODO: Stage for handling video post processing

    // Final Stage: NV12toARGB color space conversion
    cudaLaunchNV12toARGBDrv(*ppDecodedFrame, nDecodedPitch, nBytesPerSample,
                                      *ppTextureData, nTexturePitch,
                                      nWidth, nHeight, fpCudaKernel, streamID);
}

// This is the CUDA stage for Video Post Processing.  Last stage takes care of the NV12 to ARGB
void
cudaPostProcessFrame(CUdeviceptr *ppDecodedFrame, size_t nDecodedPitch, int nBytesPerSample,
                     CUarray array,
                     CUmodule cuModNV12toARGB,
                     CUfunction fpCudaKernel, CUstream streamID)
{
    uint32 nWidth  = g_pVideoDecoder->targetWidth();
    uint32 nHeight = g_pVideoDecoder->targetHeight();

    // Upload the Color Space Conversion Matrices
    if (g_bUpdateCSC)
    {
        // CCIR 601/709
        float hueColorSpaceMat[9];
        setColorSpaceMatrix(g_eColorSpace,    hueColorSpaceMat, g_nHue);
        updateConstantMemory_drvapi(cuModNV12toARGB, hueColorSpaceMat);

        if (!g_bUpdateAll)
        {
            g_bUpdateCSC = false;
        }
    }

    // TODO: Stage for handling video post processing

    // Final Stage: NV12toARGB color space conversion
    cudaLaunchNV12toARGBDrv(*ppDecodedFrame, nDecodedPitch, nBytesPerSample,
                                      g_pRgba, nWidth * 4,
                                      nWidth, nHeight, fpCudaKernel, streamID);

    CUDA_MEMCPY2D memcpy2D = { 0 };
    memcpy2D.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    memcpy2D.srcDevice = g_pRgba;
    memcpy2D.srcPitch = nWidth * 4;
    memcpy2D.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    memcpy2D.dstArray = array;
    memcpy2D.dstPitch = nWidth * 4;
    memcpy2D.WidthInBytes = nWidth * 4;
    memcpy2D.Height = nHeight;

    // clear the surface to solid white
    checkCudaErrors(cuMemcpy2D(&memcpy2D));
}

// Draw the final result on the screen
HRESULT drawScene(int field_num)
{
    HRESULT hr = S_OK;

    // init the scene
    if (g_bUseDisplay)
    {
        // render image
        g_pImageDX->render(field_num);
    }

    hr = g_pSwapChain->Present(g_bUseVsync ? DXGI_SWAP_EFFECT_SEQUENTIAL : DXGI_SWAP_EFFECT_DISCARD, 0);

    return S_OK;
}

// Release all previously initd objects
HRESULT cleanup(bool bDestroyContext)
{
    if (fpWriteYUV != NULL)
    {
        fflush(fpWriteYUV);
        fclose(fpWriteYUV);
        fpWriteYUV = NULL;
    }

    if (fpRefYUV != NULL)
    {
        fflush(fpRefYUV);
        fclose(fpRefYUV);
        fpRefYUV = NULL;
    }

    if (bDestroyContext)
    {
        // Attach the CUDA Context (so we may properly free memroy)
        checkCudaErrors(cuCtxPushCurrent(g_oContext));

        if (g_pInteropFrame[0])
        {
            checkCudaErrors(cuMemFree(g_pInteropFrame[0]));
        }

        if (g_pInteropFrame[1])
        {
            checkCudaErrors(cuMemFree(g_pInteropFrame[1]));
        }

        if (g_pInteropFrame[2])
        {
            checkCudaErrors(cuMemFree(g_pInteropFrame[2]));
        }

        if (g_pRgba) {
            checkCudaErrors(cuMemFree(g_pRgba));
        }

        // Detach from the Current thread
        checkCudaErrors(cuCtxPopCurrent(NULL));
    }

    if (g_pImageDX)
    {
        delete g_pImageDX;
        g_pImageDX = NULL;
    }

    freeCudaResources(bDestroyContext);

    // destroy the D3D device
    if (g_pD3DDevice)
    {
        g_pD3DDevice->Release();
        g_pD3DDevice = NULL;
    }

    if (g_pContext) {
        g_pContext->Release();
        g_pContext = NULL;
    }

    if (g_pSwapChain) {
        g_pSwapChain->Release();
        g_pSwapChain = NULL;
    }

    return S_OK;
}

// Launches the CUDA kernels to fill in the texture data
void renderVideoFrame(HWND hWnd, bool bUseInterop)
{
    static unsigned int nRepeatFrame = 0;
    int repeatFactor = g_iRepeatFactor;
    int bIsProgressive = 1, bFPSComputed = 0;
    bool bFramesDecoded = false;

    if (0 != g_pFrameQueue)
    {
        // if not running, we simply don't copy new frames from the decoder
        if (g_bRunning)
        {
            bFramesDecoded = copyDecodedFrameToTexture(nRepeatFrame, true, &bIsProgressive);
        }
    }
    else
    {
        return;
    }

    if (bFramesDecoded)
    {
        while (repeatFactor-- > 0)
        {
            // draw the scene using the copied textures
            if (g_bUseDisplay && bUseInterop)
            {
                for (unsigned int i = 0; i < nRepeatFrame; i++) {
                    drawScene(i);
                    if (!repeatFactor)
                    {
                        computeFPS(hWnd, bUseInterop);
                    }
                }

                bFPSComputed = 1;
            }
        }

        // Pass the Windows handle to show Frame Rate on the window title
        if (!bFPSComputed)
        {
            computeFPS(hWnd, bUseInterop);
        }
    }

    if (bFramesDecoded && g_bFrameStep)
    {
        if (g_bRunning)
        {
            g_bRunning = false;
        }
    }
}

// The window's message handler
static LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
        case WM_KEYDOWN:
            switch (wParam)
            {
                    // use ESC to quit application
                case VK_ESCAPE:
                    {
                        g_bDone = true;
                        PostQuitMessage(0);
                        return 0;
                    }
                    break;

                    // use space to pause playback
                case VK_SPACE:
                    {
                        g_bRunning = !g_bRunning;
                    }
                    break;
            }

            break;

        case WM_DESTROY:
            g_bDone = true;
            PostQuitMessage(0);
            return 0;

        case WM_PAINT:
            ValidateRect(hWnd, NULL);
            return 0;
    }

    return DefWindowProc(hWnd, msg, wParam, lParam);
}

