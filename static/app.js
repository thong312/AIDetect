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

function switchMode(mode){
	document.getElementById('btn-register').classList.remove('active');
	document.getElementById('btn-recognize').classList.remove('active');
	document.getElementById('section-register').classList.add('hidden');
	document.getElementById('section-recognize').classList.add('hidden');
	if(mode==='register'){
		document.getElementById('btn-register').classList.add('active');
		document.getElementById('section-register').classList.remove('hidden');
		startCamera('video-reg');
	} else {
		document.getElementById('btn-recognize').classList.add('active');
		document.getElementById('section-recognize').classList.remove('hidden');
		startCamera('video-rec');
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
	list.innerHTML = `<div class="result-item"><b>Tuổi:</b> ${data.info.age}<br><b>Giới:</b> ${data.info.gender}<br><b>Cảm xúc:</b> ${data.info.emotion}</div>`;
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

