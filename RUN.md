# راهنمای اجرا

دستورها را در PowerShell و از ریشهٔ مخزن اجرا کنید.

## ۱. آماده‌سازی رایانه

به Python 3.11 نیاز دارید. محیط و وابستگی‌ها را نصب کنید:

```powershell
py -3.11 -m venv RESULTS/runtime
& .\RESULTS\runtime\Scripts\python.exe -m pip install -r .\PC\requirements.txt
& .\RESULTS\runtime\Scripts\python.exe -m pip check
```

برای اجرای مجموعهٔ کامل تست‌ها، PyTorch نیز در همین محیط لازم است:

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

## ۳. اجرای FPGA

برای TinyEye ابتدا بستهٔ استقرار تأییدشده را با مسیرهای ثبت‌شده در `RESULTS/models/demo_deployment.json` بازیابی کنید. پیش‌بررسی فایل‌ها و سپس شناسایی برد:

```powershell
.\PC\run.cmd demo --model tiny --transport jtag --check --offline
.\PC\run.cmd demo --model tiny --transport jtag --check
```

وقتی پیش‌بررسی متصل `READY` گزارش کرد و پیکربندی مرجع UART روی برد فعال بود، اجرای زیر موقتاً FPGA را برنامه‌ریزی می‌کند. با `Q` یا `Esc` خارج شوید تا برنامه پیکربندی مرجع را بازیابی کند.

```powershell
.\PC\run.cmd demo --model tiny --transport jtag --camera 0
```

مسیر مستقیم UART برای بردی که از قبل firmware سازگار دارد:

```powershell
.\PC\run.cmd fpga --port COM3 --camera 0
```

در فرمان UART، `COM3` را با پورت واقعی برد جایگزین کنید.

برای ساخت و برنامه‌ریزی، برد Spartan-6، Xilinx ISE 14.7 و Vivado/XSDB 2023.2 لازم است. پورت سریال را از Device Manager پیدا کنید و به‌جای `COM3` پورت خودتان را بنویسید.

```powershell
.\PC\run.cmd pretrained prepare
.\PC\run.cmd build --network pretrained --pe-count 2
.\PC\run.cmd program
.\PC\run.cmd fpga --port COM3
```

اجرای CPU به برد نیاز ندارد. مسیر Pretrained FPGA از UART با نرخ ۱٬۰۰۰٬۰۰۰ baud استفاده می‌کند.

## پوشه‌های خروجی

مدل‌ها و فایل‌های محلی در `RESULTS` هستند. فایل‌های اجرا و گزارش‌ها در `RESULTS/runs` و `RESULTS/logs` ذخیره می‌شوند؛ این خروجی‌ها وارد Git نمی‌شوند.
