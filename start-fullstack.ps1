# Ultra Doc-Intelligence - Full Stack Startup Script
# Starts all services: FastAPI (Python), Node.js Gateway, React Frontend

Write-Host "🚀 Starting Ultra Doc-Intelligence Full Stack..." -ForegroundColor Cyan

# Check if we're in the right directory
$srcPath = $PSScriptRoot
if (-not $srcPath) {
    $srcPath = Get-Location
}

# Start FastAPI backend (Python ML Services)
Write-Host "`n📡 Starting FastAPI ML Services (port 8000)..." -ForegroundColor Yellow
$fastapi = Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$srcPath'; .\.venv\Scripts\Activate.ps1; uvicorn app.main:app --reload --host 0.0.0.0 --port 8000" -PassThru

Start-Sleep -Seconds 2

# Start Node.js API Gateway
Write-Host "🔌 Starting Node.js API Gateway (port 3001)..." -ForegroundColor Green
$nodejs = Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$srcPath\backend'; npm start" -PassThru

Start-Sleep -Seconds 2

# Start React Frontend
Write-Host "⚛️ Starting React Frontend (port 3000)..." -ForegroundColor Blue
$react = Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$srcPath\frontend'; npm run dev" -PassThru

Write-Host "`n✅ All services started!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host "🖥️  React UI:        http://localhost:3000" -ForegroundColor White
Write-Host "🔌 Node.js Gateway: http://localhost:3001" -ForegroundColor White
Write-Host "📡 FastAPI Docs:    http://localhost:8000/docs" -ForegroundColor White
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host "`nPress any key to stop all services..." -ForegroundColor Gray

$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Stop all services
Write-Host "`n⏹️ Stopping services..." -ForegroundColor Yellow
Stop-Process -Id $fastapi.Id -ErrorAction SilentlyContinue
Stop-Process -Id $nodejs.Id -ErrorAction SilentlyContinue
Stop-Process -Id $react.Id -ErrorAction SilentlyContinue

Write-Host "✅ All services stopped." -ForegroundColor Green
