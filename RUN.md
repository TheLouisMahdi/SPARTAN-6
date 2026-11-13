# راهنمای اجرا

دستورها را در PowerShell و از ریشهٔ مخزن اجرا کنید.

## ۱. آماده‌سازی رایانه

به Python 3.11 نیاز دارید. محیط و وابستگی‌ها را نصب کنید:

```powershell
py -3.11 -m venv RESULTS/runtime
& .\RESULTS\runtime\Scripts\python.exe -m pip install -r .\PC\requirements.txt
& .\RESULTS\runtime\Scripts\python.exe -m pip check
```

برای اجرای مجموعهٔ کامل تست‌ها، PyTorch در همین محیط و Icarus Verilog 13.0 در مسیر `RESULTS/tools/iverilog/ucrt64/bin` لازم‌اند:

```powershell
& .\RESULTS\runtime\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple 'torch==2.7.1+cpu'
.\PC\run.cmd test-all
```

برای اجرای دوربین، فایل مدل Face Landmarker را دریافت کنید و در این مسیر ذخیره کنید:

```text
RESULTS/models/face_landmarker.task
```

برای دریافت آن از PowerShell:

```powershell
Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task' -OutFile .\RESULTS\models\face_landmarker.task
```

## ۲. نمونهٔ وب‌کم روی رایانه

از ریشهٔ مخزن اجرا کنید:

```powershell
.\PC\run.cmd live --camera 0
```

پنجره برای هر فریم وضعیت `LEFT` و `RIGHT` را جداگانه نشان می‌دهد. مدل هر دو چشم را از یک فریم در یک نوبت پردازش می‌کند. اگر یکی از چشم‌ها پیدا نشود، وضعیت هر دو `UNAVAILABLE` است. برای خروج `Q` یا `Esc` را بزنید. اگر دوربین باز نشد، شمارهٔ `1` را امتحان کنید.

برای اجرای همان نمونه با محاسبات ثابت‌نقطه، `--precision fixed` را به دستور اضافه کنید.

## ۳. اجرای FPGA از طریق UART

برای ساخت تصویر FPGA به Xilinx ISE 14.7 و برای بارگذاری موقت از طریق کابل JTAG به Vivado/XSDB 2023.2 نیاز است. اگر برد از قبل firmware سازگار دارد، مستقیم به اجرای وب‌کم بروید.

```powershell
.\PC\run.cmd build --network pretrained --pe-count 2
if ($LASTEXITCODE -ne 0) { throw 'FPGA build failed' }
.\PC\run.cmd program
if ($LASTEXITCODE -ne 0) { throw 'FPGA programming failed' }
```

پورت مبدل UART را در Device Manager یا با فرمان زیر پیدا کنید:

```powershell
& .\RESULTS\runtime\Scripts\python.exe -m serial.tools.list_ports -v
```

`COM3` در مثال زیر یک نمونه است. آن را با پورت برد خود جایگزین کنید. مسیر Pretrained با نرخ ۱٬۰۰۰٬۰۰۰ baud کار می‌کند.

```powershell
.\PC\run.cmd fpga --port COM3 --camera 0 --pipeline sequential
```

## ۴. اجرای اختیاری TinyEye از طریق JTAG

برای TinyEye فایل‌های بستهٔ استقرار تأییدشده را در `RESULTS/deployment` قرار دهید و مسیرهایشان را در `RESULTS/models/demo_deployment.json` ثبت کنید. پیش‌بررسی فایل‌ها و سپس شناسایی برد:

```powershell
.\PC\run.cmd demo --model tiny --transport jtag --check --offline
.\PC\run.cmd demo --model tiny --transport jtag --check
```

وقتی پیش‌بررسی متصل `READY` گزارش کرد و پیکربندی مرجع UART روی برد فعال بود، اجرای زیر موقتاً FPGA را برنامه‌ریزی می‌کند. با `Q` یا `Esc` خارج شوید تا برنامه پیکربندی مرجع را بازیابی کند.

```powershell
.\PC\run.cmd demo --model tiny --transport jtag --camera 0
```

بستهٔ TinyEye شامل checkpoint آموزش‌دیده، تصویر FPGA، فایل بازگشت و شواهد اعتبارسنجی است. این فایل‌ها باید جداگانه تهیه شوند.

## پوشه‌های خروجی

مدل‌ها و فایل‌های محلی در `RESULTS` هستند. فایل‌های اجرا و گزارش‌ها در `RESULTS/runs` و `RESULTS/logs` ذخیره می‌شوند؛ این خروجی‌ها وارد Git نمی‌شوند.
