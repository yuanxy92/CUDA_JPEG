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

#ifndef NV_IMAGE_DX
#define NV_IMAGE_DX

#include "dynlink_cuda.h" // <cuda.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#endif

const int Format2Bpp[] = { 1, 4, 0 };

class ImageDX
{
    public:
        enum PixelFormat
        {
            LUMINANCE_PIXEL_FORMAT,
            BGRA_PIXEL_FORMAT,
            UNKNOWN_PIXEL_FORMAT
        };

        ImageDX(ID3D11Device *pDeviceD3D, ID3D11DeviceContext *pContext,
                IDXGISwapChain *pSwapChain,
                unsigned int nDispWidth, unsigned int nDispHeight,
                unsigned int nTexWidth,  unsigned int nTexHeight,
                bool bVsync,
                PixelFormat ePixelFormat = BGRA_PIXEL_FORMAT);

        // Destructor
        ~ImageDX();

        void
        registerAsCudaResource(int field_num);

        void
        unregisterAsCudaResource(int field_num);

        bool
        isCudaResource()
        const;

        void
        setCUDAcontext(CUcontext oContext);

        void
        setCUDAdevice(CUdevice oDevice);

        int Bpp()
        {
            return Format2Bpp[(int)e_PixFmt_];
        }

        // Map this image's DX surface into CUDA memory space.
        // Parameters:
        //      ppImageData - point to point to image data. On return this
        //          pointer references the mapped data.
        //      pImagePitch - pointer to image pitch. On return of this
        //          pointer contains the pitch of the mapped image surface.
        // Note:
        //      This method will fail, if this image is not a registered CUDA resource.
        void
        map(CUarray *pBackBufferArray, int active_field = 0);

        void
        unmap(int active_field = 0);

        // Clear the image.
        // Parameters:
        //      nClearColor - the luminance value to clear the image to. Default is white.
        // Note:
        //      This method will not work if this image is not registered as a CUDA resource at the
        //      time of this call.
        void
        clear(unsigned char nClearColor = 0xff);

        unsigned int
        width()
        const;

        unsigned int
        height()
        const;

        void
        render(int active_field = 0)
        const;

    private:
        static
        DXGI_FORMAT
        d3dFormat(PixelFormat ePixelFormat);

        ID3D11Device *pDeviceD3D_;
        ID3D11DeviceContext *pContext_;
        IDXGISwapChain *pSwapChain_;
        ID3D11Texture2D *pTexture_[3];
        ID3D11Texture2D *pBackBuffer;
        CUgraphicsResource aCudaResource_[3];

        unsigned int nWidth_;
        unsigned int nHeight_;
        unsigned int nTexWidth_;
        unsigned int nTexHeight_;
        PixelFormat e_PixFmt_;

        bool bIsCudaResource_;
        bool bVsync_;

        CUcontext oContext_;
        CUdevice  oDevice_;
};

#endif // NV_IMAGE_DX