@echo off
rem Lamina's draft layer for Qwen3.8-Flash-Next (the engine's --mtp folder): four files, 0.8 GB, from
rem https://huggingface.co/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP at a pinned revision, each checked by its SHA-256.
rem
rem   download-draft-layer.bat FOLDER      then in the config's "args":  "--mtp", "FOLDER"
rem
rem A download that broke off goes on from where it stopped. HF_ENDPOINT names a mirror. It needs curl.exe and
rem certutil.exe, which Windows 10 (1803) and newer have.
setlocal
if "%~1"=="" goto usage
if not "%~2"=="" goto usage
set "DEST=%~f1"
set "REV=ff2f70927caafe90ebd1f2ea1c08a9e014a65ce4"
set "BASE=https://huggingface.co"
if defined HF_ENDPOINT set "BASE=%HF_ENDPOINT%"
set "URL=%BASE%/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP/resolve/%REV%"
where curl >nul 2>nul
if errorlevel 1 goto nocurl
if not exist "%DEST%\" mkdir "%DEST%"
if not exist "%DEST%\" goto nofolder
call :get dense.txt 8773c81ebb0986e37fe94a8a9933e87be48b1fabc6889a0106bcdab179d1c2ac
if errorlevel 1 exit /b 1
call :get draft_vocab.bin 369151522226a5edaa5f12cfd1e2ae7db8f4fbdbd222f3dcf327dced9597fb25
if errorlevel 1 exit /b 1
call :get dense.bin a46561c5d3e76e78eebb9b54bbf4ae87ba1f99a62cad85562cbb516141cd5e47
if errorlevel 1 exit /b 1
call :get experts.bin 09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f
if errorlevel 1 exit /b 1
echo.
echo The draft layer is in "%DEST%"
echo Name the folder in the config's "args":  "--mtp", "%DEST:\=/%"
exit /b 0

:usage
echo usage: %~nx0 FOLDER
echo Downloads Lamina's draft layer for Qwen3.8-Flash-Next (0.8 GB) into FOLDER; then name the folder in the
echo config's "args":  "--mtp", "FOLDER"
exit /b 2

:nocurl
echo curl.exe was not found: Windows 10 (1803) and newer have it
exit /b 1

:nofolder
echo the folder "%DEST%" could not be made
exit /b 1

:get
rem %1: the file's name, %2: its SHA-256
set "FILE=%DEST%\%1"
if not exist "%FILE%" goto part
call :sha "%FILE%"
if /i "%HASH%"=="%2" goto there
echo "%FILE%" is not the file expected (its SHA-256 differs): move it away and run this again
exit /b 1
:there
echo %1: there already, checked
exit /b 0
:part
if not exist "%FILE%.part" goto fetch
call :sha "%FILE%.part"
if /i "%HASH%"=="%2" goto keep
:fetch
echo %1 ...
curl -L --fail --retry 5 --retry-delay 5 -C - -o "%FILE%.part" "%URL%/%1"
if errorlevel 1 goto failed
call :sha "%FILE%.part"
if /i "%HASH%"=="%2" goto keep
echo %1: the download is not the file expected (its SHA-256 differs): delete "%FILE%.part" and run this again
exit /b 1
:keep
move /y "%FILE%.part" "%FILE%" >nul
if errorlevel 1 exit /b 1
echo %1: downloaded, checked
exit /b 0
:failed
echo %1: the download failed. Run this again to go on from where it stopped; if it fails the same way, delete
echo "%FILE%.part" first
exit /b 1

:sha
rem HASH: the SHA-256 of %1 (the second line certutil prints; older Windows put spaces between its bytes)
set "HASH="
for /f "skip=1 delims=" %%h in ('certutil -hashfile %1 SHA256') do if not defined HASH set "HASH=%%h"
if not defined HASH exit /b 1
set "HASH=%HASH: =%"
exit /b 0
