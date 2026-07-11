const { chromium } = require('playwright-core');
(async () => {
  const browser = await chromium.launch({ executablePath: '/root/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome' });
  const page = await browser.newPage({ viewport: { width: 980, height: 700 } });
  const errs = [], cons = [];
  page.on('pageerror', e => errs.push('PAGEERROR: ' + e.message));
  page.on('console', m => { if (m.type()==='error'||m.type()==='warning') cons.push(m.type().toUpperCase()+': '+m.text()); });
  await page.goto('http://localhost:3031/2', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);
  const info = await page.evaluate(() => {
    const out = {};
    out.path = location.pathname + location.hash;
    const pages = Array.from(document.querySelectorAll('.slidev-page'));
    out.pageCount = pages.length;
    out.pages = pages.map((p,i) => {
      const cs = getComputedStyle(p);
      const r = p.getBoundingClientRect();
      return { i, cls: p.className, display: cs.display, visibility: cs.visibility, opacity: cs.opacity, w: Math.round(r.width), h: Math.round(r.height), rectTop: Math.round(r.top), text: (p.innerText||'').slice(0,40).replace(/\n/g,' '), hasMuLu: p.innerHTML.includes('目录') };
    });
    // the active one: largest visible rect / non-zero size & display!=none
    const vis = pages.find(p => getComputedStyle(p).display!=='none' && p.getBoundingClientRect().width>100);
    out.activeIdx = vis ? pages.indexOf(vis) : -1;
    if (vis) {
      out.activeText = (vis.innerText||'').slice(0,200);
      const h1 = vis.querySelector('h1');
      if (h1){const cs=getComputedStyle(h1);out.h1={text:h1.innerText,display:cs.display,visibility:cs.visibility,opacity:cs.opacity,color:cs.color,fontSize:cs.fontSize,w:h1.offsetWidth,h:h1.offsetHeight,mt:h1.style.marginTop||cs.marginTop};}
      const flex = vis.querySelector('div[style*="flex"]');
      if(flex){const cs=getComputedStyle(flex);out.flex={display:cs.display,visibility:cs.visibility,opacity:cs.opacity,w:flex.offsetWidth,h:flex.offsetHeight,children:flex.children.length};} else out.flex='NO flex';
      const lay = vis.querySelector('.slidev-layout')||vis;
      out.layBg=getComputedStyle(lay).backgroundImage.slice(0,50);
      out.layColor=getComputedStyle(lay).color;
      out.layDisplay=getComputedStyle(lay).display;
      out.layHeight=getComputedStyle(lay).height;
      out.layOverflow=getComputedStyle(lay).overflow;
    }
    return out;
  });
  console.log(JSON.stringify(info, null, 2));
  console.log('--- pageerrors ---'); console.log(errs.join('\n') || 'none');
  console.log('--- console err/warn ---'); console.log(cons.join('\n') || 'none');
  await browser.close();
})();
