@echo off
chcp 65001 >nul
setlocal DisableDelayedExpansion
pushd "%~dp0"
if errorlevel 1 goto folder_error
if not exist "%~dp0.venv\Scripts\python.exe" goto python_error
"%~dp0.venv\Scripts\python.exe" "%~dp0windows_launcher.py" --create-shortcut %*
if errorlevel 1 goto launch_error
echo Ярлык готов. Теперь программу можно запускать с рабочего стола.
pause
popd
exit /b 0

:python_error
echo Не найден рабочий Python: "%~dp0.venv\Scripts\python.exe"
goto launch_error

:folder_error
echo Не удалось открыть папку проекта: "%~dp0"
pause
exit /b 1

:launch_error
echo Не удалось создать ярлык. Прочитайте причину выше.
pause
popd
exit /b 1
