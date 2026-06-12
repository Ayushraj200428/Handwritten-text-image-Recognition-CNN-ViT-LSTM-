const statusEl = document.getElementById('status');
const statusBadgeEl = document.getElementById('statusBadge');
const outputTextEl = document.getElementById('outputText');
const confidenceEl = document.getElementById('confidenceValue');
const confidenceBarEl = document.getElementById('confidenceBar');
const timestampEl = document.getElementById('timestamp');
const statusTextEl = document.getElementById('statusText');
const processingEl = document.getElementById('processing');
const alertEl = document.getElementById('alert');
const dropZoneEl = document.getElementById('dropZone');
const fileInputEl = document.getElementById('fileInput');
const cameraInputEl = document.getElementById('cameraInput');
const previewWrapEl = document.getElementById('previewWrap');
const previewImgEl = document.getElementById('previewImg');
const removeImageBtnEl = document.getElementById('removeImageBtn');
const browseBtnEl = document.getElementById('browseBtn');
const cameraBtnEl = document.getElementById('cameraBtn');
const predictBtnEl = document.getElementById('predictBtn');
const uploadHintEl = document.getElementById('uploadHint');
const navUploadBtnEl = document.getElementById('navUploadBtn');
const heroUploadBtnEl = document.getElementById('heroUploadBtn');
const heroDetectBtnEl = document.getElementById('heroDetectBtn');
const cursorGlowEl = document.getElementById('cursorGlow');

const ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
let uploadDataUrl = '';

function setBadge(text, mode = 'idle') {
	statusBadgeEl.textContent = text;
	statusBadgeEl.className = `chip ${mode}`;
}

function setAlert(message) {
	if (!message) {
		alertEl.classList.add('hidden');
		alertEl.textContent = '';
		return;
	}
	alertEl.textContent = message;
	alertEl.classList.remove('hidden');
}

function setProcessing(active) {
	processingEl.classList.toggle('active', active);
}

function setConfidence(value) {
	const conf = Math.max(0, Math.min(100, value));
	confidenceEl.textContent = `${conf}%`;
	confidenceBarEl.style.width = `${conf}%`;
}

function resetOutput() {
	outputTextEl.textContent = 'No text detected yet.';
	setConfidence(0);
	timestampEl.textContent = '--';
	statusTextEl.textContent = 'Ready';
	setBadge('Idle', 'idle');
}

async function checkHealth() {
	try {
		const res = await fetch('/health');
		if (!res.ok) throw new Error();
		statusEl.textContent = 'Backend Online';
		statusEl.style.color = '#9ae6b4';
	} catch {
		statusEl.textContent = 'Backend Offline';
		statusEl.style.color = '#fca5a5';
		setAlert('Backend is offline. Please start the Flask server.');
	}
}

function setPreview(dataUrl) {
	uploadDataUrl = dataUrl;
	previewImgEl.src = dataUrl;
	previewWrapEl.classList.remove('hidden');
	predictBtnEl.disabled = false;
	uploadHintEl.textContent = 'Image ready. Click Run to analyze.';
	setAlert('');
}

function clearPreview() {
	uploadDataUrl = '';
	previewImgEl.src = '';
	previewWrapEl.classList.add('hidden');
	predictBtnEl.disabled = true;
	uploadHintEl.textContent = 'Select or drop an image to begin.';
	fileInputEl.value = '';
	cameraInputEl.value = '';
}

function handleFile(file) {
	if (!file) return;
	if (!ALLOWED_TYPES.includes(file.type)) {
		setAlert('Invalid image format. Please upload png, jpg, jpeg, or webp.');
		return;
	}
	const reader = new FileReader();
	reader.onload = () => setPreview(String(reader.result || ''));
	reader.readAsDataURL(file);
}

async function runPrediction() {
	if (!uploadDataUrl) {
		setAlert('Please upload an image before running OCR.');
		return;
	}

	setProcessing(true);
	setAlert('');
	setBadge('Processing', 'idle');
	statusTextEl.textContent = 'Processing';

	try {
		const res = await fetch('/predict', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ file: uploadDataUrl }),
		});

		const data = await res.json();
		if (!data.success) {
			throw new Error(data.error || 'Prediction failed');
		}

		const text = String(data.text || '').trim();
		const conf = Math.round(Number(data.confidence || 0) * 100);

		outputTextEl.textContent = text || 'No text detected.';
		setConfidence(conf);
		timestampEl.textContent = data.timestamp || new Date().toLocaleTimeString();
		statusTextEl.textContent = data.status || 'Completed';
		setBadge(text ? 'Completed' : 'No Text', 'idle');

		if (!text) {
			setAlert(data.message || 'No text region detected in the image.');
		}
	} catch (error) {
		setBadge('Error', 'idle');
		statusTextEl.textContent = 'Error';
		setAlert(error.message || 'Server error. Please try again.');
	} finally {
		setProcessing(false);
	}
}

dropZoneEl.addEventListener('click', () => fileInputEl.click());
browseBtnEl.addEventListener('click', () => fileInputEl.click());
cameraBtnEl.addEventListener('click', () => cameraInputEl.click());
navUploadBtnEl.addEventListener('click', () => fileInputEl.click());
heroUploadBtnEl.addEventListener('click', () => fileInputEl.click());
heroDetectBtnEl.addEventListener('click', () => document.getElementById('ocr').scrollIntoView());

fileInputEl.addEventListener('change', (event) => handleFile(event.target.files?.[0]));
cameraInputEl.addEventListener('change', (event) => handleFile(event.target.files?.[0]));
removeImageBtnEl.addEventListener('click', clearPreview);
predictBtnEl.addEventListener('click', runPrediction);

['dragenter', 'dragover'].forEach((eventName) => {
	dropZoneEl.addEventListener(eventName, (event) => {
		event.preventDefault();
		dropZoneEl.classList.add('dragover');
	});
});

['dragleave', 'drop'].forEach((eventName) => {
	dropZoneEl.addEventListener(eventName, (event) => {
		event.preventDefault();
		dropZoneEl.classList.remove('dragover');
	});
});

dropZoneEl.addEventListener('drop', (event) => {
	const file = event.dataTransfer?.files?.[0];
	handleFile(file);
});

document.addEventListener('mousemove', (event) => {
	if (!cursorGlowEl) return;
	cursorGlowEl.style.left = `${event.clientX}px`;
	cursorGlowEl.style.top = `${event.clientY}px`;
});

const observer = new IntersectionObserver(
	(entries) => {
		entries.forEach((entry) => {
			if (entry.isIntersecting) {
				entry.target.classList.add('in-view');
			}
		});
	},
	{ threshold: 0.15 }
);

document.querySelectorAll('.reveal').forEach((el) => observer.observe(el));

resetOutput();
checkHealth();
setInterval(checkHealth, 10000); 

