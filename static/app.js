// APP JS: camera, capture, UI interactions
const startCamera = async (videoId) => {
	try {
		const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
		const video = document.getElementById(videoId);
		if (!video) return;
		video.srcObject = stream;
		await video.play().catch(()=>{});
	} catch (e) {
		console.error('Camera error', e);
		alert('Không thể truy cập camera.');
	}
};

const stopCamera = (videoId) => {
	const video = document.getElementById(videoId);
	if (!video || !video.srcObject) return;
	video.srcObject.getTracks().forEach(track => track.stop());
	video.srcObject = null;
};

function toggleCamera(videoId, btnId){
	const btn = document.getElementById(btnId);
	const video = document.getElementById(videoId);
	
	if(video.srcObject){
		stopCamera(videoId);
		btn.innerText = '📹 Bật Camera';
		btn.classList.remove('camera-on');
	} else {
		startCamera(videoId);
		btn.innerText = '📹 Tắt Camera';
		btn.classList.add('camera-on');
	}
};

function switchMode(mode){
	document.getElementById('btn-register').classList.remove('active');
	document.getElementById('btn-recognize').classList.remove('active');
	document.getElementById('btn-analyze').classList.remove('active');
	document.getElementById('section-register').classList.add('hidden');
	document.getElementById('section-recognize').classList.add('hidden');
	document.getElementById('section-analyze').classList.add('hidden');
	if(mode==='register'){
		document.getElementById('btn-register').classList.add('active');
		document.getElementById('section-register').classList.remove('hidden');
		stopAutoRecognize();
		stopVideoAnalysis();
		startCamera('video-reg');
		document.getElementById('btn-camera-reg').innerText = '📹 Tắt Camera';
		document.getElementById('btn-camera-reg').classList.add('camera-on');
	} else if(mode==='analyze'){
		document.getElementById('btn-analyze').classList.add('active');
		document.getElementById('section-analyze').classList.remove('hidden');
		stopAutoRecognize();
		startCamera('video-analyze');
		document.getElementById('btn-camera-analyze').innerText = '📹 Tắt Camera';
		document.getElementById('btn-camera-analyze').classList.add('camera-on');
	} else {
		document.getElementById('btn-recognize').classList.add('active');
		document.getElementById('section-recognize').classList.remove('hidden');
		stopVideoAnalysis();
		startCamera('video-rec');
		startAutoRecognize();
		document.getElementById('btn-camera-rec').innerText = '📹 Tắt Camera';
		document.getElementById('btn-camera-rec').classList.add('camera-on');
	}
}

function updateName(name, info){
	const el = document.getElementById('name-big');
	const meta = document.getElementById('meta-small');
	el.innerText = name;
	if(info && info.age){
		meta.innerText = `Tuổi ${info.age} • ${info.gender} • ${info.emotion}`;
	} else {
		meta.innerText = 'Chưa phân tích';
	}
}

function captureToBlob(videoEl){
	const canvas = document.createElement('canvas');
	canvas.width = videoEl.videoWidth || 640;
	canvas.height = videoEl.videoHeight || 480;
	const ctx = canvas.getContext('2d');
	ctx.drawImage(videoEl,0,0,canvas.width,canvas.height);
	return new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg'));
}

async function captureRecognize(){
	const video = document.getElementById('video-rec');
	if(!video) return;
	const blob = await captureToBlob(video);
	const form = new FormData();
	form.append('file', blob, 'rec.jpg');

	const res = await fetch('/recognize',{method:'POST',body:form});
	const data = await res.json();
	updateName(data.name, data.info||{});
	const list = document.getElementById('recognize-result');
	list.innerHTML = `<div class="result-item"><b>Tên:</b> ${data.name}<br><b>Khoảng cách:</b> ${data.distance}<br><b>Cảm xúc:</b> ${data.info?.emotion||'-'}</div>`;
}

async function analyzeOnce(){
	const video = document.getElementById('video-rec');
	if(!video) return;
	const blob = await captureToBlob(video);
	const form = new FormData();
	form.append('file', blob, 'analyze.jpg');
	const res = await fetch('/analyze',{method:'POST',body:form});
	const data = await res.json();
	updateName('--', data.info||{});
	const list = document.getElementById('recognize-result');
	
	const info = data.info;
	list.innerHTML = `
		<div class="confidence-item">
			<div class="confidence-label">
				<span>Tuổi: ${info.age}</span>
				<span class="confidence-value">${info.age_confidence}%</span>
			</div>
			<div class="confidence-bar">
				<div class="confidence-fill" style="width: ${info.age_confidence}%"></div>
			</div>
		</div>
		<div class="confidence-item">
			<div class="confidence-label">
				<span>Giới: ${info.gender}</span>
				<span class="confidence-value">${info.gender_confidence}%</span>
			</div>
			<div class="confidence-bar">
				<div class="confidence-fill" style="width: ${info.gender_confidence}%"></div>
			</div>
		</div>
		<div class="confidence-item">
			<div class="confidence-label">
				<span>Cảm xúc: ${info.emotion}</span>
				<span class="confidence-value">${info.emotion_confidence}%</span>
			</div>
			<div class="confidence-bar">
				<div class="confidence-fill" style="width: ${info.emotion_confidence}%"></div>
			</div>
		</div>
	`;
}

async function captureRegister(){
	const name = document.getElementById('reg-name').value.trim();
	if(!name) return alert('Nhập tên trước.');
	const video = document.getElementById('video-reg');
	const blob = await captureToBlob(video);
	const form = new FormData();
	form.append('file', blob, 'reg.jpg');
	form.append('name', name);
	const res = await fetch('/register',{method:'POST',body:form});
	const data = await res.json();
	document.getElementById('register-result').innerHTML = `<div class="result-item">${data.message}</div>`;
}

async function registerWithFile(){
	const name = document.getElementById('reg-name').value.trim();
	if(!name) return alert('Nhập tên trước.');
	const fileInput = document.getElementById('reg-file');
	if(!fileInput.files.length) return alert('Chọn ảnh trước.');
	const file = fileInput.files[0];
	const form = new FormData();
	form.append('file', file);
	form.append('name', name);
	const res = await fetch('/register',{method:'POST',body:form});
	const data = await res.json();
	document.getElementById('register-result').innerHTML = `<div class="result-item">${data.message}</div>`;
}

// Preview image when file is selected
document.addEventListener('DOMContentLoaded', ()=>{
	const fileInput = document.getElementById('reg-file');
	if(fileInput){
		fileInput.addEventListener('change', (e)=>{
			const file = e.target.files[0];
			if(file){
				const reader = new FileReader();
				reader.onload = (event)=>{
					const preview = document.getElementById('preview-reg');
					preview.src = event.target.result;
					preview.classList.remove('hidden');
				};
				reader.readAsDataURL(file);
			}
		});
	}
});

// Video Real-time Analysis
let videoAnalysisInterval = null;
let isAnalyzing = false;

function toggleVideoAnalysis(){
	if(isAnalyzing){
		stopVideoAnalysis();
	} else {
		startVideoAnalysis();
	}
}

function startVideoAnalysis(){
	if(isAnalyzing) return;
	isAnalyzing = true;
	const btn = document.getElementById('btn-analyze-toggle');
	btn.innerText = '⏹ Dừng';
	btn.classList.add('active');
	
	videoAnalysisInterval = setInterval(async ()=>{
		const video = document.getElementById('video-analyze');
		const canvas = document.getElementById('canvas-analyze');
		if(!video || !video.srcObject) return;
		try{
			// Vẽ frame video lên canvas
			const ctx = canvas.getContext('2d');
			ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
			
			// Lấy blob từ canvas
			canvas.toBlob(async (blob)=>{
				const form = new FormData();
				form.append('file', blob, 'analyze.jpg');
				const res = await fetch('/analyze',{method:'POST',body:form});
				const data = await res.json();
				updateAnalysisDisplay(data.info||{});
			}, 'image/jpeg');
		} catch(e){
			console.error('Video analysis error:', e);
		}
	}, 1000);
}

function stopVideoAnalysis(){
	if(videoAnalysisInterval){
		clearInterval(videoAnalysisInterval);
		videoAnalysisInterval = null;
	}
	isAnalyzing = false;
	const btn = document.getElementById('btn-analyze-toggle');
	btn.innerText = '▶ Bắt đầu';
	btn.classList.remove('active');
	
	// Xóa canvas
	const canvas = document.getElementById('canvas-analyze');
	const ctx = canvas.getContext('2d');
	ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function updateAnalysisDisplay(info){
	// Cập nhật overlay text
	const emotion = document.getElementById('analyze-emotion');
	const meta = document.getElementById('analyze-meta');
	emotion.innerText = info.emotion || '--';
	meta.innerText = `Tuổi ${info.age} • ${info.gender}`;
	
	// Cập nhật kết quả chi tiết
	const list = document.getElementById('analyze-result');
	
	list.innerHTML = `
		<div class="confidence-item">
			<div class="confidence-label">
				<span>Tuổi: ${info.age}</span>
				<span class="confidence-value">${info.age_confidence||0}%</span>
			</div>
			<div class="confidence-bar">
				<div class="confidence-fill" style="width: ${info.age_confidence||0}%"></div>
			</div>
		</div>
		<div class="confidence-item">
			<div class="confidence-label">
				<span>Giới: ${info.gender}</span>
				<span class="confidence-value">${info.gender_confidence||0}%</span>
			</div>
			<div class="confidence-bar">
				<div class="confidence-fill" style="width: ${info.gender_confidence||0}%"></div>
			</div>
		</div>
		<div class="confidence-item">
			<div class="confidence-label">
				<span>Cảm xúc: ${info.emotion}</span>
				<span class="confidence-value">${info.emotion_confidence||0}%</span>
			</div>
			<div class="confidence-bar">
				<div class="confidence-fill" style="width: ${info.emotion_confidence||0}%"></div>
			</div>
		</div>
	`;
}

// Clock
function startClock(){
	const el = document.getElementById('clock');
	setInterval(()=>{
		const d=new Date();
		el.innerText = d.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
	},1000);
}

// Init default
window.addEventListener('load',()=>{
	startClock();
	// start camera for recognize by default
	switchMode('recognize');
});

