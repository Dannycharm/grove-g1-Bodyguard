"""Encode offline PIL RGB frames using installed libav libraries, without installs."""
from pathlib import Path
import subprocess,tempfile,shlex

def encode_mp4(frames,output,fps=25):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if not frames:raise ValueError('No video frames')
    width,height=frames[0].size
    source=Path(__file__).with_name('encode_rgb_mp4.c')
    with tempfile.TemporaryDirectory(prefix='escort_video_encoder_') as tmp:
        binary=Path(tmp)/'encode_rgb_mp4'
        try:flags=shlex.split(subprocess.check_output(['pkg-config','--cflags','--libs','libavcodec','libavformat','libavutil','libswscale'],text=True))
        except (FileNotFoundError,subprocess.CalledProcessError) as exc:raise RuntimeError('Requires existing cc, pkg-config and libavcodec/libavformat/libavutil/libswscale development files; no dependencies were installed') from exc
        subprocess.run(['cc',str(source),'-o',str(binary),*flags],check=True)
        with output.with_suffix('.encoding.log').open('x') as log:
            proc=subprocess.Popen([str(binary),str(output),str(width),str(height),str(fps)],stdin=subprocess.PIPE,stderr=log)
            try:
                for frame in frames:
                    if frame.size!=(width,height):raise ValueError('Frame dimensions changed')
                    proc.stdin.write(frame.convert('RGB').tobytes())
                proc.stdin.close()
                if proc.wait():raise RuntimeError('Video encoding failed; inspect encoding log')
            finally:
                if proc.poll() is None:proc.kill();proc.wait()
