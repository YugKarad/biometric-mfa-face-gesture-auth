const video = document.getElementById("video");
const startBtn = document.getElementById("startCamera");
const stopBtn = document.getElementById("stopCamera");
const statusText = document.getElementById("status");

let stream = null;

startBtn.addEventListener("click", async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        });
        video.srcObject = stream;
        statusText.textContent = " ✅ Camera active";
    } catch (err) {
        statusText.textContent = " ❌ Camera access denied";
        console.error(err);
    }
});

stopBtn.addEventListener("click", () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        statusText.textContent = "Camera stopped";
    }
});
