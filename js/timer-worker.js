// Timers inside a worker are not throttled like the page's rAF / setInterval
// when the tab is in the background, so posture checks keep running.
let id = null;

onmessage = (e) => {
  const { cmd, interval } = e.data;
  clearInterval(id);
  if (cmd === "start") id = setInterval(() => postMessage("tick"), interval);
};
