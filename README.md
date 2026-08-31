# RunPod TTS Worker v2

Current worker version: `2026-08-31-flowtts-restore`

รุ่นนี้กู้กลับมาใช้ทางเดิมที่เสียงไทยเคยชัด: `biodatlab/thonburian-tts` + `FlowTTS` + checkpoint/vocab จาก `gostiktok14sroy/gostts` ไม่ใช้ `f5-tts_infer-cli` เป็นทางหลัก

Worker นี้รับ payload จาก AI Voice Cloning Local และคืนเสียง WAV แบบ base64

รองรับ:

- `jaitts` สำหรับภาษาไทย
- `cosyvoice3` สำหรับภาษาจีนและภาษาอื่น
- โหลดโมเดลครั้งแรกแบบ lazy และเก็บ cache ใน Network Volume
- ใช้ `/runsync` ตาม backend ปัจจุบัน
- มี `.runpod/hub.json` และ `.runpod/tests.json` ตาม worker template รุ่นปัจจุบัน

## 1. นำโฟลเดอร์ขึ้น GitHub

สร้าง repository ใหม่ แล้วอัปโหลดไฟล์ทั้งหมดที่อยู่ข้างในโฟลเดอร์ `runpod-worker-v2` โดยให้ `handler.py`, `requirements.txt` และ `Dockerfile` อยู่ที่ root ของ repository ไม่ควรมีโฟลเดอร์ `runpod-worker-v2` ครอบอีกชั้น

ห้ามใส่ RunPod API key หรือ Hugging Face token ลงในไฟล์

## 2. Deploy บน RunPod

1. ไปที่ Serverless > New Endpoint
2. เลือก Import from Git Repository หรือเชื่อม GitHub repository นี้
3. เลือก Endpoint Type เป็น `Queue`
4. ถ้าไฟล์อยู่ใน root ให้ตั้ง Dockerfile Path เป็น `Dockerfile`
5. เลือก GPU เริ่มต้นเป็น RTX 4090 หรือ A40
6. ตั้ง Active workers = 0, Max workers = 1
7. ตั้ง Execution timeout = 900 วินาที
8. ถ้ามี Network Volume ให้ mount ที่ `/runpod-volume`
9. เพิ่ม Environment Variable ชื่อ `HF_TOKEN` หากโมเดลต้องใช้ token
10. Deploy แล้วรอ build เสร็จ

## 3. ทดสอบ Endpoint

ในแท็บ Requests ส่ง:

```json
{
  "input": {
    "healthcheck": true
  }
}
```

ผลที่ถูกต้องต้องมี `status: READY` และ `cuda: true`

## 4. เชื่อมกับแอป

ในหน้า Settings ของแอป:

- TTS Runtime: `RunPod`
- Endpoint ID: คัดลอกจาก RunPod
- API Key: สร้างจาก RunPod Settings > API Keys
- Custom URL: ปล่อยว่างเพื่อให้แอปสร้าง URL `/runsync` อัตโนมัติ
- Timeout: `900`

กดบันทึก แล้วลองเจนเสียงเพียงหนึ่ง dialogue ก่อน

## หมายเหตุ

- งานแรกช้ากว่างานถัดไปเพราะต้องดาวน์โหลดและโหลดโมเดล (cold start)
- Network Volume ช่วยไม่ต้องดาวน์โหลดโมเดลใหม่ทุกครั้ง
- Payload `/runsync` จำกัดขนาด จึงควรใช้ reference audio สั้นประมาณ 5-15 วินาที
- ถ้า build CosyVoice dependency ไม่ผ่าน ให้เริ่มทดสอบ JaiTTS ก่อน แล้วดู Build Logs บรรทัดแรกที่ error
