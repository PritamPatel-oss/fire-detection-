// ============================================
//   FIREGUARD AI - MAIN JAVASCRIPT
// ============================================

let currentImageBase64 = null;
let cameraStream = null;

// ---- Mode Switching ----
function switchMode(mode) {
  const uploadSection = document.getElementById('uploadSection');
  const cameraSection = document.getElementById('cameraSection');
  const uploadBtn = document.getElementById('uploadBtn');
  const cameraBtn = document.getElementById('cameraBtn');

  hideResult();

  if (mode === 'upload') {
    uploadSection.style.display = 'block';
    cameraSection.style.display = 'none';
    uploadBtn.classList.add('active');
    cameraBtn.classList.remove('active');
    stopCamera();
  } else {
    uploadSection.style.display = 'none';
    cameraSection.style.display = 'block';
    cameraBtn.classList.add('active');
    uploadBtn.classList.remove('active');
  }
}

// ---- Handle File Upload ----
function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function (e) {
    currentImageBase64 = e.target.result;
    const preview = document.getElementById('uploadPreview');
    preview.innerHTML = `<img src="${currentImageBase64}" alt="Uploaded image"/>`;
    document.getElementById('uploadDetectBtn').style.display = 'block';
    hideResult();
  };
  reader.readAsDataURL(file);
}

// Drag & Drop
const dropZone = document.getElementById('dropZone');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.style.borderColor = '#ff4500'; });
dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = ''; });
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.style.borderColor = '';
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    const fakeEvent = { target: { files: [file] } };
    handleFileUpload(fakeEvent);
  }
});

// ---- Camera Functions ----
async function startCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
    document.getElementById('cameraFeed').srcObject = cameraStream;
  } catch (err) {
    alert('❌ Camera access denied. Please allow camera permissions.');
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
    document.getElementById('cameraFeed').srcObject = null;
  }
}

function captureAndDetect() {
  const video = document.getElementById('cameraFeed');
  const canvas = document.getElementById('captureCanvas');

  if (!cameraStream) {
    alert('⚠️ Please start the camera first!');
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);

  currentImageBase64 = canvas.toDataURL('image/jpeg');

  const preview = document.getElementById('cameraPreview');
  preview.innerHTML = `<img src="${currentImageBase64}" alt="Captured frame"/>`;

  detectFire('camera');
}

// ---- Fire Detection API Call ----
async function detectFire(source) {
  if (!currentImageBase64) {
    alert('⚠️ No image selected!');
    return;
  }

  showLoader();
  hideResult();

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: currentImageBase64 })
    });

    const data = await response.json();

    if (data.error) {
      hideLoader();
      alert('❌ Error: ' + data.error);
      return;
    }

    hideLoader();
    showResult(data.fire_detected, data.confidence);

  } catch (err) {
    hideLoader();
    alert('❌ Failed to connect to server. Make sure Flask is running.');
  }
}

// ---- Show Result ----
function showResult(fireDetected, confidence) {
  const section = document.getElementById('resultSection');
  const icon = document.getElementById('resultIcon');
  const title = document.getElementById('resultTitle');
  const fill = document.getElementById('confidenceFill');
  const confText = document.getElementById('confidenceText');
  const message = document.getElementById('resultMessage');

  section.style.display = 'block';

  if (fireDetected) {
    icon.textContent = '🔥';
    title.textContent = 'Fire Detected!';
    title.className = 'result-title fire';
    fill.className = 'confidence-fill fire';
    message.textContent = '⚠️ WARNING: Fire presence detected! Please take immediate action and evacuate if necessary.';
    document.body.classList.add('fire-alert');
    setTimeout(() => document.body.classList.remove('fire-alert'), 2500);
    playAlarm();
  } else {
    icon.textContent = '✅';
    title.textContent = 'No Fire Detected';
    title.className = 'result-title safe';
    fill.className = 'confidence-fill safe';
    message.textContent = '✔️ The area appears safe. No fire detected in the provided image.';
    stopAlarm();
  }

  // Animate confidence bar
  setTimeout(() => {
    fill.style.width = confidence + '%';
  }, 100);

  confText.textContent = confidence.toFixed(1) + '%';

  // Scroll to result
  section.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ---- Loader ----
function showLoader() { document.getElementById('loader').style.display = 'block'; }
function hideLoader() { document.getElementById('loader').style.display = 'none'; }
function hideResult() { document.getElementById('resultSection').style.display = 'none'; }

// ---- Alarm ----
function playAlarm() {
  const alarm = document.getElementById('alarmSound');
  alarm.currentTime = 0;
  alarm.play().catch(() => {});
  document.getElementById('stopAlarmBtn').style.display = 'block';
}

function stopAlarm() {
  const alarm = document.getElementById('alarmSound');
  alarm.pause();
  alarm.currentTime = 0;
  document.getElementById('stopAlarmBtn').style.display = 'none';
}