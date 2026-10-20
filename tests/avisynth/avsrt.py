"""Minimal AviSynth+ runtime for tests, over the C API with ctypes.

- `Env.eval(script)` runs a script and returns a Clip (raises AvsError with the
  script error message).
- `Env.source(frames, fmt)` builds a clip whose frames come from numpy arrays
  (a C filter implemented by a Python callback), so AviSynth and VapourSynth
  can be fed bit-identical input.
- `Clip.frame(n)` returns the active pixels of every plane as numpy arrays plus
  the frame properties.

Planes are returned in vszip's logical order: Y,U,V / R,G,B, then alpha.
"""

import ctypes as C
import numpy as np

PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A = 1, 2, 4, 16
PLANAR_R, PLANAR_G, PLANAR_B = 32, 64, 128


class AvsError(Exception):
    pass


class _D(C.Union):
    _fields_ = [("clip", C.c_void_p), ("boolean", C.c_char), ("integer", C.c_int),
                ("floating_pt", C.c_float), ("string", C.c_char_p), ("array", C.c_void_p),
                ("longlong", C.c_int64), ("double_pt", C.c_double)]


class Value(C.Structure):
    _fields_ = [("type", C.c_short), ("array_size", C.c_short), ("d", _D)]


class VideoInfo(C.Structure):
    _fields_ = [("width", C.c_int), ("height", C.c_int), ("fps_numerator", C.c_uint),
                ("fps_denominator", C.c_uint), ("num_frames", C.c_int), ("pixel_type", C.c_int),
                ("audio_samples_per_second", C.c_int), ("sample_type", C.c_int),
                ("num_audio_samples", C.c_int64), ("nchannels", C.c_int), ("image_type", C.c_int)]


GET_FRAME = C.CFUNCTYPE(C.c_void_p, C.c_void_p, C.c_int)


class FilterInfo(C.Structure):
    _fields_ = [("child", C.c_void_p), ("vi", VideoInfo), ("env", C.c_void_p),
                ("get_frame", GET_FRAME), ("get_parity", C.c_void_p), ("get_audio", C.c_void_p),
                ("set_cache_hints", C.c_void_p), ("free_filter", C.c_void_p),
                ("error", C.c_char_p), ("user_data", C.c_void_p)]


assert C.sizeof(Value) == 16 and C.sizeof(VideoInfo) == 48

_lib = None


def lib():
    global _lib
    if _lib is None:
        L = C.CDLL("AviSynth.dll")
        sig = {
            "avs_create_script_environment": (C.c_void_p, [C.c_int]),
            "avs_delete_script_environment": (None, [C.c_void_p]),
            "avs_get_error": (C.c_char_p, [C.c_void_p]),
            "avs_invoke": (Value, [C.c_void_p, C.c_char_p, Value, C.c_void_p]),
            "avs_release_value": (None, [Value]),
            "avs_take_clip": (C.c_void_p, [Value, C.c_void_p]),
            "avs_set_to_clip": (None, [C.POINTER(Value), C.c_void_p]),
            "avs_release_clip": (None, [C.c_void_p]),
            "avs_get_video_info": (C.POINTER(VideoInfo), [C.c_void_p]),
            "avs_get_frame": (C.c_void_p, [C.c_void_p, C.c_int]),
            "avs_clip_get_error": (C.c_char_p, [C.c_void_p]),
            "avs_release_video_frame": (None, [C.c_void_p]),
            "avs_get_read_ptr_p": (C.c_void_p, [C.c_void_p, C.c_int]),
            "avs_get_write_ptr_p": (C.c_void_p, [C.c_void_p, C.c_int]),
            "avs_get_pitch_p": (C.c_int, [C.c_void_p, C.c_int]),
            "avs_get_row_size_p": (C.c_int, [C.c_void_p, C.c_int]),
            "avs_get_height_p": (C.c_int, [C.c_void_p, C.c_int]),
            "avs_new_video_frame_a": (C.c_void_p, [C.c_void_p, C.POINTER(VideoInfo), C.c_int]),
            "avs_new_c_filter": (C.c_void_p, [C.c_void_p, C.POINTER(C.POINTER(FilterInfo)), Value, C.c_int]),
            "avs_set_var": (C.c_int, [C.c_void_p, C.c_char_p, Value]),
            "avs_save_string": (C.c_char_p, [C.c_void_p, C.c_char_p, C.c_int]),
            "avs_get_frame_props_ro": (C.c_void_p, [C.c_void_p, C.c_void_p]),
            "avs_prop_num_keys": (C.c_int, [C.c_void_p, C.c_void_p]),
            "avs_prop_get_key": (C.c_char_p, [C.c_void_p, C.c_void_p, C.c_int]),
            "avs_prop_num_elements": (C.c_int, [C.c_void_p, C.c_void_p, C.c_char_p]),
            "avs_prop_get_type": (C.c_char, [C.c_void_p, C.c_void_p, C.c_char_p]),
            "avs_prop_get_int": (C.c_int64, [C.c_void_p, C.c_void_p, C.c_char_p, C.c_int, C.POINTER(C.c_int)]),
            "avs_prop_get_float": (C.c_double, [C.c_void_p, C.c_void_p, C.c_char_p, C.c_int, C.POINTER(C.c_int)]),
            "avs_prop_get_data": (C.c_void_p, [C.c_void_p, C.c_void_p, C.c_char_p, C.c_int, C.POINTER(C.c_int)]),
            "avs_prop_get_data_size": (C.c_int, [C.c_void_p, C.c_void_p, C.c_char_p, C.c_int, C.POINTER(C.c_int)]),
        }
        for name, (res, args) in sig.items():
            f = getattr(L, name)
            f.restype = res
            f.argtypes = args
        _lib = L
    return _lib


# pixel_type bits (avisynth_c.h)
CS_YUVA, CS_BGR, CS_YUV, CS_INTERLEAVED, CS_PLANAR = 1 << 27, 1 << 28, 1 << 29, 1 << 30, 1 << 31
BITS = {0: 8, 5: 10, 6: 12, 7: 14, 1: 16, 2: 32}


def describe(pixel_type):
    """(family, bits, has_alpha, ssw, ssh) of a planar pixel_type."""
    pt = pixel_type & 0xFFFFFFFF
    if not pt & CS_PLANAR:
        raise AvsError("not planar")
    bits = BITS[(pt >> 16) & 7]
    if pt & CS_BGR:
        return ("rgb", bits, bool(pt & 2), 0, 0)
    if pt & CS_INTERLEAVED and pt & CS_YUV:
        return ("gray", bits, False, 0, 0)
    ssw = (((pt >> 0) & 7) + 1) & 3
    ssh = (((pt >> 8) & 7) + 1) & 3
    return ("yuv", bits, bool(pt & CS_YUVA), ssw, ssh)


def plane_ids(pixel_type):
    fam, _, alpha, _, _ = describe(pixel_type)
    ids = {"gray": [PLANAR_Y], "yuv": [PLANAR_Y, PLANAR_U, PLANAR_V], "rgb": [PLANAR_R, PLANAR_G, PLANAR_B]}[fam]
    return ids + ([PLANAR_A] if alpha else [])


def dtype_of(bits):
    return np.uint8 if bits == 8 else np.float32 if bits == 32 else np.uint16


class Frame:
    def __init__(self, planes, props):
        self.planes = planes
        self.props = props

    def __getitem__(self, i):
        return self.planes[i]


class Clip:
    def __init__(self, env, ptr):
        self.env = env
        self.ptr = ptr
        vi = lib().avs_get_video_info(ptr).contents
        self.width, self.height = vi.width, vi.height
        self.num_frames = vi.num_frames
        self.pixel_type = vi.pixel_type
        self.fps = (vi.fps_numerator, vi.fps_denominator)
        self.image_type = vi.image_type
        if vi.width:
            self.family, self.bits, self.has_alpha, self.ssw, self.ssh = describe(vi.pixel_type)

    def frame(self, n):
        L = lib()
        f = L.avs_get_frame(self.ptr, n)
        if not f:
            err = L.avs_clip_get_error(self.ptr)
            raise AvsError(err.decode() if err else "avs_get_frame failed")
        try:
            dt = dtype_of(self.bits)
            planes = []
            for pid in plane_ids(self.pixel_type):
                p = L.avs_get_read_ptr_p(f, pid)
                pitch = L.avs_get_pitch_p(f, pid)
                row = L.avs_get_row_size_p(f, pid)
                h = L.avs_get_height_p(f, pid)
                buf = (C.c_ubyte * (pitch * (h - 1) + row)).from_address(p)
                raw = np.frombuffer(buf, dtype=np.uint8)
                arr = np.lib.stride_tricks.as_strided(raw, shape=(h, row), strides=(pitch, 1))
                # Always copy: when pitch == row size the strided view is already
                # contiguous and ascontiguousarray would alias the host buffer,
                # which AviSynth recycles once the frame is released.
                planes.append(np.array(arr, copy=True).view(dt))
            props = self.env._props(f)
        finally:
            L.avs_release_video_frame(f)
        return Frame(planes, props)

    def frames(self, ns=None):
        return [self.frame(n) for n in (range(self.num_frames) if ns is None else ns)]

    def __del__(self):
        if self.ptr and self.env.ptr:
            lib().avs_release_clip(self.ptr)
            self.ptr = None


class Env:
    def __init__(self, plugins=(), version=11):
        L = lib()
        self.ptr = L.avs_create_script_environment(version)
        if not self.ptr or not C.cast(self.ptr, C.POINTER(C.c_void_p))[0]:
            raise AvsError(f"cannot create an AviSynth environment for interface {version}")
        self._keep = []  # callbacks and arrays that must outlive the clips
        for p in plugins:
            self.eval(f'LoadPlugin("{p}")', want_clip=False)

    def close(self):
        if self.ptr:
            lib().avs_delete_script_environment(self.ptr)
            self.ptr = None

    def _err(self, v):
        msg = v.d.string.decode("utf-8", "replace") if v.d.string else "unknown error"
        return AvsError(msg)

    def eval(self, script, want_clip=True):
        L = lib()
        arg = Value()
        arg.type = ord("s")
        arg.array_size = 0
        arg.d.string = L.avs_save_string(self.ptr, script.encode(), -1)
        v = L.avs_invoke(self.ptr, b"Eval", arg, None)
        if v.type == ord("e"):
            raise self._err(v)
        try:
            if not want_clip:
                return None
            if v.type != ord("c"):
                raise AvsError(f"script returned type {chr(v.type)}, not a clip")
            return Clip(self, L.avs_take_clip(v, self.ptr))
        finally:
            L.avs_release_value(v)

    def set_clip(self, name, clip):
        L = lib()
        v = Value()
        L.avs_set_to_clip(C.byref(v), clip.ptr)
        L.avs_set_var(self.ptr, L.avs_save_string(self.ptr, name.encode(), -1), v)
        L.avs_release_value(v)

    def source(self, frames, pixel_type, fps=(24, 1), name=None):
        """Clip of `frames` (list of per-plane 2D arrays, logical order)."""
        L = lib()
        h, w = frames[0][0].shape
        blank = self.eval(f'BlankClip(length={len(frames)}, width={w}, height={h}, pixel_type="{pixel_type}", '
                          f'fps={fps[0]}, fps_denominator={fps[1]})')
        child = Value()
        L.avs_set_to_clip(C.byref(child), blank.ptr)
        fi = C.POINTER(FilterInfo)()
        clip_ptr = L.avs_new_c_filter(self.ptr, C.byref(fi), child, 1)
        L.avs_release_value(child)
        ids = plane_ids(fi.contents.vi.pixel_type)
        data = [[np.ascontiguousarray(p) for p in f] for f in frames]

        def get_frame(fi_ptr, n):
            info = C.cast(fi_ptr, C.POINTER(FilterInfo)).contents
            n = min(max(n, 0), len(data) - 1)
            # avs_new_video_frame_p_a with a NULL prop_src crashes the host.
            out = L.avs_new_video_frame_a(info.env, C.byref(info.vi), 64)
            for pid, arr in zip(ids, data[n]):
                dst = L.avs_get_write_ptr_p(out, pid)
                pitch = L.avs_get_pitch_p(out, pid)
                row = L.avs_get_row_size_p(out, pid)
                b = arr.view(np.uint8)
                assert b.shape[1] == row, (b.shape, row)
                for y in range(b.shape[0]):
                    C.memmove(dst + y * pitch, b[y].ctypes.data, row)
            return out

        cb = GET_FRAME(get_frame)
        fi.contents.get_frame = cb
        self._keep += [cb, data, blank]
        clip = Clip(self, clip_ptr)
        if name:
            self.set_clip(name, clip)
        return clip

    def _props(self, f):
        L = lib()
        m = L.avs_get_frame_props_ro(self.ptr, f)
        out = {}
        err = C.c_int(0)
        for i in range(L.avs_prop_num_keys(self.ptr, m)):
            k = L.avs_prop_get_key(self.ptr, m, i)
            t = L.avs_prop_get_type(self.ptr, m, k).decode()
            n = L.avs_prop_num_elements(self.ptr, m, k)
            if t == "i":
                vals = [L.avs_prop_get_int(self.ptr, m, k, j, C.byref(err)) for j in range(n)]
            elif t == "f":
                vals = [L.avs_prop_get_float(self.ptr, m, k, j, C.byref(err)) for j in range(n)]
            elif t == "s":
                vals = []
                for j in range(n):
                    p = L.avs_prop_get_data(self.ptr, m, k, j, C.byref(err))
                    sz = L.avs_prop_get_data_size(self.ptr, m, k, j, C.byref(err))
                    vals.append(C.string_at(p, sz) if sz > 0 else b"")
            else:
                vals = [f"<{t}>"] * n
            out[k.decode()] = vals[0] if n == 1 else vals
        return out
