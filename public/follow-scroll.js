// Follow the document, not an inner transcript panel. Reading older text wins.
export function followLatest(content, active) {
  const scroller = () => document.scrollingElement;
  let following = true, frame = 0, lastTop = window.scrollY, lastHeight = scroller().scrollHeight, touchY = null;
  const nearBottom = () => scroller().scrollHeight - window.innerHeight - window.scrollY <= 48;
  function pause() { following = false; }
  function schedule() {
    if (frame || !following || !active()) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      if (!following || !active()) return;
      window.scrollTo({ top: scroller().scrollHeight, behavior: 'instant' });
      lastTop = window.scrollY;
      lastHeight = scroller().scrollHeight;
    });
  }
  window.addEventListener('scroll', () => {
    const top = window.scrollY, height = scroller().scrollHeight;
    // Layout shrinkage can clamp scrollY; that is not a request to read upward.
    if (top < lastTop - 1 && height === lastHeight) following = false;
    else if (top > lastTop + 1 && nearBottom()) { following = true; schedule(); }
    lastTop = top; lastHeight = height;
  }, { passive: true });
  window.addEventListener('wheel', event => { if (event.deltaY < 0) pause(); }, { passive: true });
  window.addEventListener('touchstart', event => { touchY = event.touches[0]?.clientY ?? null; }, { passive: true });
  window.addEventListener('touchmove', event => {
    const y = event.touches[0]?.clientY;
    if (touchY !== null && y > touchY + 2) pause();
    touchY = y ?? null;
  }, { passive: true });
  window.addEventListener('keydown', event => {
    if (event.target.closest?.('input, textarea, select, [contenteditable="true"]')) return;
    if (['ArrowUp', 'PageUp', 'Home'].includes(event.key) || (event.key === ' ' && event.shiftKey)) pause();
  });
  // Resize handles text wrapping, images/fonts and completed summary blocks.
  new ResizeObserver(schedule).observe(content);
  new MutationObserver(schedule).observe(content, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ['hidden'] });
  window.addEventListener('resize', schedule);
  return { reset() { following = true; lastTop = window.scrollY; lastHeight = scroller().scrollHeight; }, update: schedule };
}
