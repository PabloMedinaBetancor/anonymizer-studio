const CODEC_MIME = {
  h264: 'video/mp4; codecs="avc1.42E01E"',
  hevc: 'video/mp4; codecs="hvc1.1.6.L93.B0"',
  vp8: 'video/webm; codecs="vp8"',
  vp9: 'video/webm; codecs="vp9"',
  av1: 'video/mp4; codecs="av01.0.05M.08"',
};

export function browserCanPlay(clip) {
  const mime = CODEC_MIME[clip.codec];
  return Boolean(mime) && document.createElement("video").canPlayType(mime) !== "";
}

// One interface over a real <video> and a clock. The clock drives frame-by-frame viewing for clips the browser
// cannot decode, where each step asks the server for the frame.
export function createPlayer(onChange) {
  let clip = null;
  let video = null;
  let simTime = 0;
  let simPlaying = false;
  let simStart = 0;
  let raf = 0;
  let detach = () => {};

  const notify = () => onChange({ time: player.time(), playing: player.playing() });

  function tick(now) {
    simTime = Math.min(clip.duration_s, simTime + (now - simStart) / 1000);
    simStart = now;
    if (simTime >= clip.duration_s) simPlaying = false;
    notify();
    if (simPlaying) raf = requestAnimationFrame(tick);
  }

  const player = {
    attach(nextClip, videoEl) {
      cancelAnimationFrame(raf);
      if (video) video.pause();
      detach();
      detach = () => {};
      clip = nextClip;
      video = videoEl ?? null;
      simTime = 0;
      simPlaying = false;
      if (video) {
        video.src = nextClip.source_url;
        const names = ["timeupdate", "play", "pause", "seeked"];
        const element = video;
        for (const name of names) element.addEventListener(name, notify);
        detach = () => names.forEach((name) => element.removeEventListener(name, notify));
      }
      notify();
    },
    time: () => (video ? video.currentTime : simTime),
    playing: () => (video ? !video.paused : simPlaying),
    toggle() {
      if (!clip) return;
      if (video) {
        if (video.paused) {
          // A play() that a pause() cuts short (two quick presses of the button or of Space) rejects its promise: nobody is waiting for it
          const started = video.play();
          if (started && typeof started.catch === "function") started.catch(() => {});
        } else video.pause();
        return;
      }
      simPlaying = !simPlaying;
      if (simPlaying) {
        simStart = performance.now();
        raf = requestAnimationFrame(tick);
      }
      notify();
    },
    seek(seconds) {
      if (!clip) return;
      const t = Math.min(Math.max(0, seconds), clip.duration_s);
      if (video) video.currentTime = t;
      else simTime = t;
      notify();
    },
    step(frames) {
      if (clip) player.seek(player.time() + frames / clip.fps);
    },
    stop() {
      cancelAnimationFrame(raf);
      if (video) video.pause();
      simPlaying = false;
    },
  };
  return player;
}
