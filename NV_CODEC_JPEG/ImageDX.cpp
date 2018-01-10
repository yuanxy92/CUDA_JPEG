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

#include "ImageDX.h"

#include "dynlink_cuda.h"     // <cuda.h>
#include "dynlink_cudaD3D11.h" // <cudaD3D11.h>

#include <cassert>

#include "helper_cuda_drvapi.h"

ImageDX::ImageDX(ID3D11Device *pDeviceD3D, ID3D11DeviceContext *pContext, IDXGISwapChain *pSwapChain,
                 unsigned int nDispWidth, unsigned int nDispHeight,
                 unsigned int nTexWidth,  unsigned int nTexHeight,
                 bool bVsync,
                 PixelFormat ePixelFormat) :
    pDeviceD3D_(pDeviceD3D), pContext_(pContext), pSwapChain_(pSwapChain)
    , nWidth_(nDispWidth)
    , nHeight_(nDispHeight)
    , nTexWidth_(nTexWidth)
    , nTexHeight_(nTexHeight)
    , bVsync_(bVsync)
    , bIsCudaResource_(false)
{
    assert(0 != pDeviceD3D_);

    pSwapChain_->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBackBuffer);

    int nFrames = bVsync_ ? 3 : 1;

    pTexture_[0] = pTexture_[1] = pTexture_[2] = 0;
 
    for (int field_num = 0; field_num < nFrames; field_num++)
    {
        D3D11_TEXTURE2D_DESC desc = {
            nWidth_, nHeight_, 1, 1, d3dFormat(ePixelFormat), { 1, 0 },
            D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, 0, 0
        };
        HRESULT hResult = pDeviceD3D_->CreateTexture2D(&desc, NULL, &pTexture_[field_num]);
        assert(S_OK == hResult);
        assert(0 != pTexture_[field_num]);
        registerAsCudaResource(field_num);
    }
}

ImageDX::~ImageDX()
{
    int nFrames = bVsync_ ? 3 : 1;

    for (int field_num=0; field_num < nFrames; field_num++)
    {
        unregisterAsCudaResource(field_num);
        pTexture_[field_num]->Release();
    }

    pBackBuffer->Release();
}

void
ImageDX::registerAsCudaResource(int field_num)
{
    // register the Direct3D resources that we'll use
    checkCudaErrors(cuGraphicsD3D11RegisterResource(&aCudaResource_[field_num], pTexture_[field_num], CU_GRAPHICS_REGISTER_FLAGS_NONE));
    getLastCudaDrvErrorMsg("cudaD3D11RegisterResource (pTexture_) failed");

    bIsCudaResource_ = true;

    // we will be write directly to this 2D texture, so we must set the
    // appropriate flags (to eliminate extra copies during map, but unmap will do copies)
    checkCudaErrors(cuGraphicsResourceSetMapFlags(aCudaResource_[field_num], CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD));
}

void
ImageDX::unregisterAsCudaResource(int field_num)
{
    CUresult result = cuCtxPushCurrent(oContext_);
    checkCudaErrors(cuGraphicsUnregisterResource(aCudaResource_[field_num]));
    bIsCudaResource_ = false;
    cuCtxPopCurrent(NULL);
}

void
ImageDX::setCUDAcontext(CUcontext oContext)
{
    oContext_ = oContext;
    printf("ImageDX::CUcontext = %08x\n", (int)oContext);
}

void
ImageDX::setCUDAdevice(CUdevice oDevice)
{
    oDevice_ = oDevice;
    printf("ImageDX::CUdevice  = %08x\n", (int)oDevice);
}

bool
ImageDX::isCudaResource()
const
{
    return bIsCudaResource_;
}

void
ImageDX::map(CUarray *pBackBufferArray, int active_field)
{
    int nFrames = bVsync_ ? 3 : 1;

    checkCudaErrors(cuGraphicsMapResources(nFrames, aCudaResource_, 0));
    checkCudaErrors(cuGraphicsSubResourceGetMappedArray(pBackBufferArray, aCudaResource_[active_field], 0, 0));
    assert(0 != *pBackBufferArray);
}

void
ImageDX::unmap(int active_field)
{
    int nFrames = bVsync_ ? 3 : 1;

    checkCudaErrors(cuGraphicsUnmapResources(nFrames, aCudaResource_, 0));
}

void
ImageDX::clear(unsigned char nClearColor)
{
    // Can only be cleared if surface is a CUDA resource
    assert(bIsCudaResource_);

    int nFrames = bVsync_ ? 3 : 1;
    CUdeviceptr  pData = 0;
    size_t nSize = nWidth_ * nHeight_ * 4;

    checkCudaErrors(cuMemAlloc(&pData, nSize));
    checkCudaErrors(cuMemsetD8(pData, nClearColor, nSize));

    checkCudaErrors(cuGraphicsMapResources(nFrames, aCudaResource_, 0));

    for (int field_num=0; field_num < nFrames; field_num++)
    {
        //checkCudaErrors(cuGraphicsResourceGetMappedPointer(&pData, &nSize, aCudaResource_[field_num]));
        CUarray array;
        checkCudaErrors(cuGraphicsSubResourceGetMappedArray(&array, aCudaResource_[field_num], 0, 0));
        assert(0 != array);
 
        CUDA_MEMCPY2D memcpy2D = { 0 };
        memcpy2D.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        memcpy2D.srcDevice = pData;
        memcpy2D.srcPitch = nWidth_ * 4;
        memcpy2D.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        memcpy2D.dstArray = array;
        memcpy2D.dstPitch = nWidth_ * 4;
        memcpy2D.WidthInBytes = nWidth_ * 4;
        memcpy2D.Height = nHeight_;

        // clear the surface to solid white
        checkCudaErrors(cuMemcpy2D(&memcpy2D));
    }

    checkCudaErrors(cuGraphicsUnmapResources(nFrames, aCudaResource_, 0));

    checkCudaErrors(cuMemFree(pData));
}

unsigned int
ImageDX::width()
const
{
    return nWidth_;
}

unsigned int
ImageDX::height()
const
{
    return nHeight_;
}

void
ImageDX::render(int active_field)
const
{
    pContext_->CopyResource(pBackBuffer, pTexture_[active_field]);
}

DXGI_FORMAT
ImageDX::d3dFormat(PixelFormat ePixelFormat)
{
    switch (ePixelFormat)
    {
        case LUMINANCE_PIXEL_FORMAT:
            return DXGI_FORMAT_R8_UNORM;

        case BGRA_PIXEL_FORMAT:
            return DXGI_FORMAT_B8G8R8A8_UNORM;

        case UNKNOWN_PIXEL_FORMAT:
            assert(false);

        default:
            assert(false);
    }

    assert(false);
    return DXGI_FORMAT_UNKNOWN;
}
