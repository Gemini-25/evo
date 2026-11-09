/*
 * mf_abi.c - prints the Media Foundation ABI facts that src/evolab/app/export/mf_win.py hard-codes, taken from the
 * Windows SDK headers, as one JSON object on stdout:
 *
 *   {"vtable": {interface: {method: slot}}, "guids": {name: "xxxxxxxx-..."}, "constants": {name: value}}
 *
 * tests/test_mf_abi.py compares it with mf_win.VTABLE, mf_win.GUIDS and mf_win.CONSTANTS (design v3 D53, W6).
 * Windows CI (MSVC + Windows SDK):
 *   cl /nologo /W3 packaging\ci\mf_abi.c /Fe:build\mf_abi.exe mfuuid.lib ole32.lib
 *   build\mf_abi.exe > build\mf_abi.json
 * It also builds with mingw-w64 (x86_64-w64-mingw32-gcc packaging/ci/mf_abi.c -o mf_abi.exe -lmfuuid -lole32).
 * Nothing here calls Media Foundation, so the runner does not need the Media Foundation feature.
 * ASCII only (the repository rule: no Hangul outside strings.py).
 */
#ifndef WINVER
#define WINVER 0x0A00
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include <windows.h>
#include <initguid.h>
#include <objbase.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfobjects.h>
#include <mfreadwrite.h>
#include <mftransform.h>
/* codecapi.h declares its GUIDs through DEFINE_GUIDEX, which only ks.h defines; a plain extern declaration is enough
   here because the GOP size GUID is rebuilt from its STATIC_ value list below. */
#ifndef DEFINE_GUIDEX
#define DEFINE_GUIDEX(name) extern const GUID name
#endif
#include <codecapi.h>
#include <stddef.h>
#include <stdio.h>

#define SLOT(iface, method) ((int)(offsetof(iface##Vtbl, method) / sizeof(void *)))

#define GUID_FROM_LIST(l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8) {l, w1, w2, {b1, b2, b3, b4, b5, b6, b7, b8}}
#define GUID_FROM(list) GUID_FROM_LIST(list)
static const GUID k_gop_size = GUID_FROM(STATIC_CODECAPI_AVEncMPVGOPSize);

static int first_item;

static void open_object(const char *name)
{
    if (name != NULL) {
        printf("%s\"%s\": {", first_item ? "" : ", ", name);
    } else {
        printf("{");
    }
    first_item = 1;
}

static void close_object(void)
{
    printf("}");
    first_item = 0;
}

static void slot(const char *method, int value)
{
    printf("%s\"%s\": %d", first_item ? "" : ", ", method, value);
    first_item = 0;
}

static void guid(const char *name, const GUID *g)
{
    printf("%s\"%s\": \"%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x\"", first_item ? "" : ", ", name,
           (unsigned long)g->Data1, (unsigned)g->Data2, (unsigned)g->Data3, g->Data4[0], g->Data4[1], g->Data4[2],
           g->Data4[3], g->Data4[4], g->Data4[5], g->Data4[6], g->Data4[7]);
    first_item = 0;
}

static void constant(const char *name, unsigned long value)
{
    printf("%s\"%s\": %lu", first_item ? "" : ", ", name, value);
    first_item = 0;
}

int main(void)
{
    open_object(NULL);

    open_object("vtable");
    open_object("IUnknown");
    slot("QueryInterface", SLOT(IUnknown, QueryInterface));
    slot("AddRef", SLOT(IUnknown, AddRef));
    slot("Release", SLOT(IUnknown, Release));
    close_object();
    open_object("IMFAttributes");
    slot("GetAllocatedString", SLOT(IMFAttributes, GetAllocatedString));
    slot("SetUINT32", SLOT(IMFAttributes, SetUINT32));
    slot("SetUINT64", SLOT(IMFAttributes, SetUINT64));
    slot("SetGUID", SLOT(IMFAttributes, SetGUID));
    close_object();
    open_object("IMFMediaType");
    slot("SetUINT32", SLOT(IMFMediaType, SetUINT32));
    slot("SetUINT64", SLOT(IMFMediaType, SetUINT64));
    slot("SetGUID", SLOT(IMFMediaType, SetGUID));
    close_object();
    open_object("IMFActivate");
    slot("GetAllocatedString", SLOT(IMFActivate, GetAllocatedString));
    close_object();
    open_object("IMFMediaBuffer");
    slot("Lock", SLOT(IMFMediaBuffer, Lock));
    slot("Unlock", SLOT(IMFMediaBuffer, Unlock));
    slot("SetCurrentLength", SLOT(IMFMediaBuffer, SetCurrentLength));
    close_object();
    open_object("IMFSample");
    slot("SetSampleTime", SLOT(IMFSample, SetSampleTime));
    slot("SetSampleDuration", SLOT(IMFSample, SetSampleDuration));
    slot("AddBuffer", SLOT(IMFSample, AddBuffer));
    close_object();
    open_object("IMFSinkWriter");
    slot("AddStream", SLOT(IMFSinkWriter, AddStream));
    slot("SetInputMediaType", SLOT(IMFSinkWriter, SetInputMediaType));
    slot("BeginWriting", SLOT(IMFSinkWriter, BeginWriting));
    slot("WriteSample", SLOT(IMFSinkWriter, WriteSample));
    slot("Finalize", SLOT(IMFSinkWriter, Finalize));
    close_object();
    close_object();

    open_object("guids");
    guid("MF_MT_MAJOR_TYPE", &MF_MT_MAJOR_TYPE);
    guid("MF_MT_SUBTYPE", &MF_MT_SUBTYPE);
    guid("MF_MT_AVG_BITRATE", &MF_MT_AVG_BITRATE);
    guid("MF_MT_INTERLACE_MODE", &MF_MT_INTERLACE_MODE);
    guid("MF_MT_FRAME_SIZE", &MF_MT_FRAME_SIZE);
    guid("MF_MT_FRAME_RATE", &MF_MT_FRAME_RATE);
    guid("MF_MT_PIXEL_ASPECT_RATIO", &MF_MT_PIXEL_ASPECT_RATIO);
    guid("MF_MT_MPEG2_PROFILE", &MF_MT_MPEG2_PROFILE);
    guid("MF_MT_DEFAULT_STRIDE", &MF_MT_DEFAULT_STRIDE);
    guid("MFMediaType_Video", &MFMediaType_Video);
    guid("MFVideoFormat_H264", &MFVideoFormat_H264);
    guid("MFVideoFormat_RGB32", &MFVideoFormat_RGB32);
    guid("MFVideoFormat_NV12", &MFVideoFormat_NV12);
    guid("MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS", &MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS);
    guid("MF_SINK_WRITER_DISABLE_THROTTLING", &MF_SINK_WRITER_DISABLE_THROTTLING);
    guid("MF_TRANSCODE_CONTAINERTYPE", &MF_TRANSCODE_CONTAINERTYPE);
    guid("MFTranscodeContainerType_MPEG4", &MFTranscodeContainerType_MPEG4);
    guid("CODECAPI_AVEncMPVGOPSize", &k_gop_size);
    guid("MFT_CATEGORY_VIDEO_ENCODER", &MFT_CATEGORY_VIDEO_ENCODER);
    guid("MFT_FRIENDLY_NAME_Attribute", &MFT_FRIENDLY_NAME_Attribute);
    close_object();

    open_object("constants");
    constant("MF_VERSION", (unsigned long)MF_VERSION);
    constant("MFSTARTUP_FULL", (unsigned long)MFSTARTUP_FULL);
    constant("COINIT_MULTITHREADED", (unsigned long)COINIT_MULTITHREADED);
    constant("MFVideoInterlace_Progressive", (unsigned long)MFVideoInterlace_Progressive);
    constant("eAVEncH264VProfile_Main", (unsigned long)eAVEncH264VProfile_Main);
    constant("MFT_ENUM_FLAG_SYNCMFT", (unsigned long)MFT_ENUM_FLAG_SYNCMFT);
    constant("MFT_ENUM_FLAG_ASYNCMFT", (unsigned long)MFT_ENUM_FLAG_ASYNCMFT);
    constant("MFT_ENUM_FLAG_HARDWARE", (unsigned long)MFT_ENUM_FLAG_HARDWARE);
    constant("MFT_ENUM_FLAG_SORTANDFILTER", (unsigned long)MFT_ENUM_FLAG_SORTANDFILTER);
    constant("S_FALSE", (unsigned long)S_FALSE);
    constant("RPC_E_CHANGED_MODE", (unsigned long)RPC_E_CHANGED_MODE);
    constant("SIZEOF_GUID", (unsigned long)sizeof(GUID));
    constant("SIZEOF_MFT_REGISTER_TYPE_INFO", (unsigned long)sizeof(MFT_REGISTER_TYPE_INFO));
    close_object();

    close_object();
    printf("\n");
    return 0;
}
