const { chromium } = require('playwright-core');
(async () => {
  const browser = await chromium.launch({ executablePath: '/root/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome' });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  await page.goto('http://localhost:3031/2?print=true', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1200);
  const info = await page.evaluate(() => {
    const out = {};
    const slides = Array.from(document.querySelectorAll('[data-slidev-no]'));
    out.totalSlides = slides.length;
    out.slideNos = slides.map(s=>s.getAttribute('data-slidev-no'));
    const s2 = document.querySelector('[data-slidev-no="2"]');
    if (!s2) { out.s2 = 'NOT FOUND'; return out; }
    const cs = getComputedStyle(s2);
    const r = s2.getBoundingClientRect();
    out.s2 = { tag: s2.tagName, cls: s2.className, display: cs.display, visibility: cs.visibility, opacity: cs.opacity, w: Math.round(r.width), h: Math.round(r.height), top: Math.round(r.top), htmlLen: s2.innerHTML.length, text: (s2.innerText||'').slice(0,160) };
    const h1 = s2.querySelector('h1');
    if (h1){const h=getComputedStyle(h1);out.h1={text:h1.innerText,display:h.display,visibility:h.visibility,color:h.color,w:h1.offsetWidth,h:h1.offsetHeight};} else out.h1='no h1';
    const flex = s2.querySelector('div[style*="flex"]');
    if(flex){const f=getComputedStyle(flex);out.flex={display:f.display,visibility:f.visibility,w:flex.offsetWidth,h:flex.offsetHeight,children:flex.children.length};} else out.flex='no flex';
    // body height in print
    out.bodyH = document.body.getBoundingClientRect().height;
    return out;
  });
  console.log(JSON.stringify(info, null, 2));
  await page.pdf({ path: '_p2.pdf', width: '1920px', height: '1080px', pageRanges: '1', printBackground: true, margin:{left:0,top:0,right:0,bottom:0} });
  await browser.close();
})();
