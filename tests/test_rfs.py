import pytest
import vapoursynth as vs

from helpers import assert_same_clip, pix


def colored(core: vs.Core, color: list, length: int = 8, **kwargs) -> vs.VideoNode:
    defaults = dict(width=64, height=32, format=vs.YUV420P8, fpsnum=24, length=length)
    return core.std.BlankClip(None, color=color, **(defaults | kwargs))


A = [50, 60, 70]
B = [200, 100, 150]


def test_frame_routing(core):
    out = core.vszip.RFS(colored(core, A), colored(core, B), frames=[0, 3, 5])
    for n in range(8):
        expected = B if n in (0, 3, 5) else A
        assert [pix(out, 0, 0, plane=p, n=n) for p in range(3)] == expected


def test_planes(core):
    out = core.vszip.RFS(colored(core, A), colored(core, B), frames=[2], planes=[0])
    assert [pix(out, 0, 0, plane=p, n=2) for p in range(3)] == [B[0], A[1], A[2]]
    assert [pix(out, 0, 0, plane=p, n=1) for p in range(3)] == A


def test_planes_chroma_only(core):
    out = core.vszip.RFS(colored(core, A), colored(core, B), frames=[2], planes=[1, 2])
    assert [pix(out, 0, 0, plane=p, n=2) for p in range(3)] == [A[0], B[1], B[2]]


def test_longer_replacement_clip(core):
    out = core.vszip.RFS(colored(core, A, length=4), colored(core, B, length=9), frames=[1])
    assert out.num_frames == 4
    assert pix(out, 0, 0, n=1) == B[0]


def test_mismatch_gives_variable_clip(core):
    a = colored(core, A)
    b = colored(core, B, width=32, height=16, format=vs.YUV420P16, fpsnum=30)
    out = core.vszip.RFS(a, b, frames=[1], mismatch=True)
    assert out.width == 0 and out.height == 0
    assert out.fps == 0
    assert out.format.id == 0  # variable format
    with out.get_frame(0) as f:
        assert (f.format.id, f.width) == (vs.YUV420P8, 64)
    with out.get_frame(1) as f:
        assert (f.format.id, f.width) == (vs.YUV420P16, 32)


def test_frame_index_error(core):
    with pytest.raises(vs.Error, match=r"frame index \(8\) > last frame index \(7\)"):
        core.vszip.RFS(colored(core, A), colored(core, B), frames=[8])


def test_plane_index_error(core):
    with pytest.raises(vs.Error, match="plane index out of range"):
        core.vszip.RFS(colored(core, A), colored(core, B), frames=[0], planes=[3])


@pytest.mark.parametrize(
    ("b_kwargs", "msg"),
    [
        (dict(width=32, height=16), "Clip dimensions don't match"),
        (dict(format=vs.YUV420P16), "Clip formats don't match"),
        (dict(fpsnum=30), "Clip frame rates don't match"),
    ],
)
def test_mismatch_required_errors(core, b_kwargs, msg):
    with pytest.raises(vs.Error, match=msg):
        core.vszip.RFS(colored(core, A), colored(core, B, **b_kwargs), frames=[0])


# --- structural coverage: frame routing patterns ----------------------------


@pytest.mark.parametrize(
    "frames",
    [
        [0],            # first frame only
        [7],            # last frame only
        [0, 7],         # both boundaries, nothing in between
        [3],            # single interior frame
        [0, 1, 2, 3, 4, 5, 6, 7],  # replace every frame
        [5, 1, 1, 3],   # unsorted with a duplicate index
    ],
    ids=lambda f: "f" + "_".join(map(str, f)),
)
def test_frame_routing_patterns(core, frames):
    """The replace mask is built from the (possibly unsorted/duplicated) frames
    list; every listed index takes clipb, every other index takes clipa."""
    out = core.vszip.RFS(colored(core, A), colored(core, B), frames=frames)
    replaced = set(frames)
    for n in range(8):
        expected = B if n in replaced else A
        assert [pix(out, 0, 0, plane=p, n=n) for p in range(3)] == expected


def test_replace_all_frames_equals_clipb(core):
    """Replacing every frame yields a pixel-identical copy of clipb."""
    b = colored(core, B)
    out = core.vszip.RFS(colored(core, A), b, frames=list(range(8)))
    assert_same_clip(out, b)


def test_replace_no_frames_in_pattern_keeps_clipa(core):
    """Frames absent from the list are passed through unchanged from clipa."""
    a = colored(core, A)
    out = core.vszip.RFS(a, colored(core, B), frames=[4])
    # every frame except 4 is bit-identical to clipa
    for n in (0, 1, 2, 3, 5, 6, 7):
        assert_same_clip(out, a, n=n)


# --- structural coverage: planes subsets across subsamplings ----------------


@pytest.mark.parametrize(
    "fmt",
    [vs.YUV420P8, vs.YUV422P8, vs.YUV444P8, vs.YUV410P8, vs.YUV411P8],
)
@pytest.mark.parametrize(
    ("planes", "want"),
    [
        ([0], lambda a, b: [b[0], a[1], a[2]]),          # luma from clipb
        ([1, 2], lambda a, b: [a[0], b[1], b[2]]),       # chroma from clipb
        ([2], lambda a, b: [a[0], a[1], b[2]]),          # single chroma plane
        ([0, 1, 2], lambda a, b: list(b)),               # all planes -> full clipb
    ],
    ids=["luma", "chroma", "v_only", "all"],
)
def test_planes_subsets_across_subsampling(core, fmt, planes, want):
    """The ShufflePlanes merge path mixes selected planes from clipb with the
    rest from clipa, regardless of chroma subsampling. Selecting all planes
    skips the merge and uses clipb whole."""
    a = colored(core, A, format=fmt)
    b = colored(core, B, format=fmt)
    out = core.vszip.RFS(a, b, frames=[2], planes=planes)
    assert [pix(out, 0, 0, plane=p, n=2) for p in range(3)] == want(A, B)
    # an unreplaced frame is always full clipa
    assert [pix(out, 0, 0, plane=p, n=1) for p in range(3)] == A


def test_planes_ignored_for_gray(core):
    """With a single plane the planes argument has no merge to do; the whole
    frame is taken from clipb on replaced frames."""
    gray = dict(format=vs.GRAY8)
    a = colored(core, [50], **gray)
    b = colored(core, [200], **gray)
    out = core.vszip.RFS(a, b, frames=[2], planes=[0])
    assert pix(out, 0, 0, n=2) == 200
    assert pix(out, 0, 0, n=1) == 50


def test_planes_subset_rgb(core):
    """RGB takes the same np>1 ShufflePlanes merge path as YUV, but passes its
    own colorFamily to ShufflePlanes. Selecting plane 0 (R) from clipb keeps
    G/B from clipa; selecting all three is the whole clipb."""
    a = core.std.BlankClip(None, width=64, height=32, format=vs.RGB24, color=[10, 20, 30], length=8)
    b = core.std.BlankClip(None, width=64, height=32, format=vs.RGB24, color=[200, 150, 100], length=8)
    out = core.vszip.RFS(a, b, frames=[2], planes=[0])
    assert [pix(out, 0, 0, plane=p, n=2) for p in range(3)] == [200, 20, 30]
    assert [pix(out, 0, 0, plane=p, n=1) for p in range(3)] == [10, 20, 30]
    full = core.vszip.RFS(a, b, frames=[2], planes=[0, 1, 2])
    assert [pix(full, 0, 0, plane=p, n=2) for p in range(3)] == [200, 150, 100]


def test_planes_all_equals_no_planes(core):
    """planes=[0,1,2] is equivalent to omitting planes entirely."""
    a, b = colored(core, A), colored(core, B)
    with_planes = core.vszip.RFS(a, b, frames=[2, 5], planes=[0, 1, 2])
    without = core.vszip.RFS(a, b, frames=[2, 5])
    assert_same_clip(with_planes, without)


# --- structural coverage: differing-length replacement clips ----------------


def test_longer_replacement_clip_routing(core):
    """clipb longer than clipa: output keeps clipa's length and indexes clipb
    by the same frame number."""
    out = core.vszip.RFS(colored(core, A, length=4), colored(core, B, length=9), frames=[1, 3])
    assert out.num_frames == 4
    assert pix(out, 0, 0, n=1) == B[0]
    assert pix(out, 0, 0, n=3) == B[0]
    assert pix(out, 0, 0, n=2) == A[0]


def test_shorter_replacement_clip_within_range(core):
    """clipb shorter than clipa: replacing a frame that exists in clipb works
    normally (this exercises the FrameReuseLastOnly request pattern)."""
    out = core.vszip.RFS(colored(core, A, length=8), colored(core, B, length=3), frames=[1])
    assert out.num_frames == 8
    assert pix(out, 0, 0, n=1) == B[0]
    assert pix(out, 0, 0, n=0) == A[0]


def test_shorter_replacement_clip_beyond_range(core):
    """Replacing a frame index past clipb's end clamps to clipb's last frame
    rather than erroring (FrameReuseLastOnly)."""
    out = core.vszip.RFS(colored(core, A, length=8), colored(core, B, length=3), frames=[5])
    assert out.num_frames == 8
    assert pix(out, 0, 0, n=5) == B[0]


# --- structural coverage: mismatch=True variable-clip shapes ----------------


def test_mismatch_format_only(core):
    """Only the format differs: output format is variable but width/height/fps
    stay fixed."""
    a = colored(core, A)
    b = colored(core, B, format=vs.YUV420P16)
    out = core.vszip.RFS(a, b, frames=[1], mismatch=True)
    assert out.format.id == 0  # variable format
    assert (out.width, out.height) == (64, 32)
    assert out.fps != 0
    with out.get_frame(0) as f:
        assert f.format.id == vs.YUV420P8
    with out.get_frame(1) as f:
        assert f.format.id == vs.YUV420P16


def test_mismatch_dimensions_only(core):
    """Only the dimensions differ: width/height go variable, format/fps fixed."""
    a = colored(core, A)
    b = colored(core, B, width=32, height=16)
    out = core.vszip.RFS(a, b, frames=[1], mismatch=True)
    assert (out.width, out.height) == (0, 0)
    assert out.format.id == vs.YUV420P8
    assert out.fps != 0
    with out.get_frame(0) as f:
        assert f.width == 64
    with out.get_frame(1) as f:
        assert f.width == 32


def test_mismatch_fps_only(core):
    """Only the frame rate differs: fps goes variable, format/dimensions fixed."""
    a = colored(core, A)
    b = colored(core, B, fpsnum=30)
    out = core.vszip.RFS(a, b, frames=[1], mismatch=True)
    assert out.fps == 0
    assert out.format.id == vs.YUV420P8
    assert (out.width, out.height) == (64, 32)


def test_mismatch_true_with_matching_clips_is_normal(core):
    """mismatch=True is harmless when the clips already agree: a fixed clip."""
    out = core.vszip.RFS(colored(core, A), colored(core, B), frames=[1], mismatch=True)
    assert out.format.id == vs.YUV420P8
    assert (out.width, out.height) == (64, 32)
    assert out.fps != 0


# --- validation: every check in the Create callback -------------------------


@pytest.mark.parametrize(
    ("a_kwargs", "b_kwargs", "args", "msg"),
    [
        # frame index bounds: index must be <= last frame index (numFrames-1)
        (dict(), dict(), dict(frames=[8]), r"frame index \(8\) > last frame index \(7\)"),
        (dict(length=4), dict(length=4), dict(frames=[4]), r"frame index \(4\) > last frame index \(3\)"),
        # plane index bounds (out of range high)
        (dict(), dict(), dict(frames=[0], planes=[3]), "plane index out of range"),
        (dict(), dict(), dict(frames=[0], planes=[5]), "plane index out of range"),
        # plane index bounds (negative -> the `e < 0` branch; planes is read as
        # signed i32, so a negative value hits the same range check)
        (dict(), dict(), dict(frames=[0], planes=[-1]), "plane index out of range"),
        # mutually-exclusive: format/dims/fps must match unless mismatch=True
        (dict(), dict(width=32, height=16), dict(frames=[0]), "Clip dimensions don't match"),
        (dict(), dict(format=vs.YUV420P16), dict(frames=[0]), "Clip formats don't match"),
        (dict(), dict(fpsnum=30), dict(frames=[0]), "Clip frame rates don't match"),
    ],
)
def test_validation_errors(core, a_kwargs, b_kwargs, args, msg):
    with pytest.raises(vs.Error, match=msg):
        core.vszip.RFS(colored(core, A, **a_kwargs), colored(core, B, **b_kwargs), **args)
