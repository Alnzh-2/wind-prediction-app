/**
 * main.js — WindPred Global JavaScript
 */

// ── Sidebar ───────────────────────────────────────────────────
function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
  document.getElementById('sidebarOverlay').classList.toggle('show');
}
function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebarOverlay').classList.remove('show');
}
// Auto-close sidebar on resize to desktop
window.addEventListener('resize', () => {
  if (window.innerWidth >= 992) closeSidebar();
});

// ── Chart.js Defaults ────────────────────────────────────────
if (typeof Chart !== 'undefined') {
  Chart.defaults.color                             = '#94a3b8';
  Chart.defaults.font.family                       = 'Space Grotesk';
  Chart.defaults.font.size                         = 11;
  Chart.defaults.plugins.tooltip.backgroundColor  = '#1a2235';
  Chart.defaults.plugins.tooltip.borderColor      = 'rgba(255,255,255,0.08)';
  Chart.defaults.plugins.tooltip.borderWidth      = 1;
  Chart.defaults.plugins.tooltip.titleColor       = '#e2e8f0';
  Chart.defaults.plugins.tooltip.bodyColor        = '#94a3b8';
  Chart.defaults.plugins.tooltip.padding          = 10;
  Chart.defaults.plugins.tooltip.cornerRadius     = 8;
  Chart.defaults.plugins.legend.labels.boxWidth   = 10;
  Chart.defaults.plugins.legend.labels.padding    = 12;
}

// ── Toast Notification ────────────────────────────────────────
function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  if (!container) return;
  const colors = { success:'#22c55e', danger:'#ef4444', info:'#0ea5e9', warning:'#f59e0b' };
  const icons  = { success:'bi-check-circle-fill', danger:'bi-x-circle-fill',
                   info:'bi-info-circle-fill', warning:'bi-exclamation-triangle-fill' };
  const id = 'toast_' + Date.now();
  container.insertAdjacentHTML('beforeend', `
    <div id="${id}" class="toast align-items-center border-0 show mb-2"
         style="background:#1a2235;border:1px solid rgba(255,255,255,0.08)!important;
                min-width:240px;border-radius:10px">
      <div class="d-flex align-items-center p-3 gap-2">
        <i class="bi ${icons[type]||icons.info}" style="color:${colors[type]||colors.info}"></i>
        <div class="text-light flex-fill" style="font-size:0.83rem">${message}</div>
        <button type="button" class="btn-close btn-close-white btn-close-sm ms-2"
                onclick="this.closest('.toast').remove()"></button>
      </div>
    </div>`);
  setTimeout(() => { document.getElementById(id)?.remove(); }, 4000);
}

// ── Global Spinner ────────────────────────────────────────────
function showSpinner() { document.getElementById('globalSpinner')?.classList.remove('d-none'); }
function hideSpinner() { document.getElementById('globalSpinner')?.classList.add('d-none'); }

// ── Helpers ───────────────────────────────────────────────────
function fmtNum(v, d=4)  { return (v==null||isNaN(v)) ? '—' : parseFloat(v).toFixed(d); }
function getDayOfYear(d) {
  return Math.floor((d - new Date(d.getFullYear(),0,0)) / 86400000);
}

// Wind category helper (mirror of Python)
function windCategory(v) {
  if (v < 1.5) return { label:'Angin Tenang', color:'success', hex:'#22c55e', icon:'🌤',
    desc:'Kondisi sangat tenang, angin hampir tidak terasa. Aman untuk semua aktivitas.' };
  if (v < 3.0) return { label:'Angin Ringan', color:'info',    hex:'#0ea5e9', icon:'🌬',
    desc:'Angin terasa di wajah, daun bergerak halus. Cocok untuk aktivitas outdoor.' };
  if (v < 5.0) return { label:'Angin Sedang', color:'warning', hex:'#f59e0b', icon:'💨',
    desc:'Daun dan ranting kecil terus bergerak. Bendera ringan berkibar.' };
  return       { label:'Angin Kencang',color:'danger',  hex:'#ef4444', icon:'🌪',
    desc:'Cabang pohon besar bergerak. Waspadai aktivitas di ketinggian.' };
}

function beaufortScale(v) {
  if (v < 0.3) return 'Calm (Beaufort 0)';
  if (v < 1.6) return 'Light Air (Beaufort 1)';
  if (v < 3.4) return 'Light Breeze (Beaufort 2)';
  if (v < 5.5) return 'Gentle Breeze (Beaufort 3)';
  if (v < 8.0) return 'Moderate Breeze (Beaufort 4)';
  return 'Fresh Breeze (Beaufort 5+)';
}
