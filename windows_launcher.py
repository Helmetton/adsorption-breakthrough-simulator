"""Windows launcher only. No application, scientific or ML state is changed."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener
import webbrowser
import uuid


ROOT = Path(__file__).resolve().parent


def same_path(first, second):
    return os.path.normcase(os.path.abspath(first)) == os.path.normcase(os.path.abspath(second))


def streamlit_script(arguments, cwd):
    """Match the actual -m streamlit run command, not a substring or port."""
    if len(arguments) < 5 or arguments[1:4] != ["-m", "streamlit", "run"]:
        return None
    script = Path(arguments[4])
    if not script.is_absolute():
        if not cwd:
            return None
        script = Path(cwd) / script
    return script


class WindowsProcesses:
    """Read-only process identity and IPv4 listener ownership, using Windows APIs."""

    def __init__(self):
        if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise RuntimeError("Этот launcher рассчитан на текущее 64-битное окружение Windows.")
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.ntdll = ctypes.WinDLL("ntdll")
        self.shell = ctypes.WinDLL("shell32", use_last_error=True)
        self.ip = ctypes.WinDLL("iphlpapi")
        self.kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        self.kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.ReadProcessMemory.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
        self.kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                        wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        self.kernel.IsWow64Process.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
        self.kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        self.kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.GenerateConsoleCtrlEvent.argtypes = [wintypes.DWORD, wintypes.DWORD]
        self.ntdll.NtQueryInformationProcess.argtypes = [wintypes.HANDLE, wintypes.ULONG, ctypes.c_void_p,
                                                        wintypes.ULONG, ctypes.POINTER(wintypes.ULONG)]
        self.ntdll.NtQueryInformationProcess.restype = wintypes.LONG
        self.shell.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
        self.shell.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
        self.kernel.LocalFree.argtypes = [ctypes.c_void_p]
        self.kernel.LocalFree.restype = ctypes.c_void_p

    def _read(self, handle, address, object_type):
        value = object_type()
        read = ctypes.c_size_t()
        if not self.kernel.ReadProcessMemory(handle, address, ctypes.byref(value), ctypes.sizeof(value), ctypes.byref(read)):
            raise OSError("Не удалось прочитать сведения процесса.")
        if read.value != ctypes.sizeof(value):
            raise OSError("Неполные сведения процесса.")
        return value

    def _unicode(self, handle, address):
        class UnicodeString(ctypes.Structure):
            _fields_ = [("Length", wintypes.USHORT), ("MaximumLength", wintypes.USHORT),
                        ("Buffer", ctypes.c_void_p)]
        descriptor = self._read(handle, address, UnicodeString)
        if not descriptor.Buffer or not descriptor.Length or descriptor.Length > 65534:
            raise OSError("Строка идентификации процесса недоступна.")
        data = self._read(handle, descriptor.Buffer, ctypes.c_char * descriptor.Length)
        return bytes(data).decode("utf-16-le")

    def identity(self, pid):
        handle = self.kernel.OpenProcess(0x0400 | 0x0010, False, pid)
        if not handle:
            return None
        try:
            wow64 = wintypes.BOOL()
            if not self.kernel.IsWow64Process(handle, ctypes.byref(wow64)) or wow64.value:
                return None
            image = ctypes.create_unicode_buffer(32768)
            size = wintypes.DWORD(len(image))
            if not self.kernel.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(size)):
                return None
            # Native 64-bit PROCESS_BASIC_INFORMATION, PEB and process parameters.
            basic = (ctypes.c_void_p * 6)()
            returned = wintypes.ULONG()
            if self.ntdll.NtQueryInformationProcess(handle, 0, ctypes.byref(basic), ctypes.sizeof(basic), ctypes.byref(returned)):
                return None
            parameters = self._read(handle, basic[1] + 0x20, ctypes.c_void_p).value
            if not parameters:
                return None
            command = self._unicode(handle, parameters + 0x70)
            cwd = self._unicode(handle, parameters + 0x38)
            argc = ctypes.c_int()
            argv = self.shell.CommandLineToArgvW(command, ctypes.byref(argc))
            if not argv:
                return None
            try:
                arguments = [argv[index] for index in range(argc.value)]
            finally:
                self.kernel.LocalFree(ctypes.cast(argv, ctypes.c_void_p))
            stamps = [wintypes.FILETIME() for _ in range(4)]
            if not self.kernel.GetProcessTimes(handle, *(ctypes.byref(stamp) for stamp in stamps)):
                return None
            created = stamps[0].dwHighDateTime << 32 | stamps[0].dwLowDateTime
            return {"pid": pid, "parent_pid": int(basic[5] or 0), "created": created,
                    "executable": image.value, "arguments": arguments, "cwd": cwd}
        except (OSError, ValueError, TypeError):
            return None
        finally:
            self.kernel.CloseHandle(handle)

    def processes(self):
        class Entry(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                        ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                        ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                        ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
                        ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]
        self.kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
        self.kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
        snapshot = self.kernel.CreateToolhelp32Snapshot(2, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            raise OSError("Windows не предоставила список процессов.")
        try:
            entry = Entry()
            entry.dwSize = ctypes.sizeof(entry)
            more = self.kernel.Process32FirstW(snapshot, ctypes.byref(entry))
            while more:
                if entry.szExeFile.lower() in {"python.exe", "pythonw.exe"}:
                    item = self.identity(entry.th32ProcessID)
                    if item:
                        yield item
                more = self.kernel.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            self.kernel.CloseHandle(snapshot)

    def listeners(self):
        size = wintypes.DWORD()
        function = self.ip.GetExtendedTcpTable
        function.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), wintypes.BOOL,
                             wintypes.ULONG, ctypes.c_int, wintypes.ULONG]
        function.restype = wintypes.DWORD
        status = function(None, ctypes.byref(size), False, socket.AF_INET, 5, 0)
        if status not in (0, 122):
            raise OSError(f"Windows не предоставила владельцев портов (код {status}).")
        for _ in range(3):
            buffer = ctypes.create_string_buffer(size.value)
            status = function(buffer, ctypes.byref(size), False, socket.AF_INET, 5, 0)
            if status == 122:
                continue
            if status:
                raise OSError(f"Не удалось проверить владельцев портов (код {status}).")
            count = wintypes.DWORD.from_buffer(buffer).value
            rows = []
            for index in range(count):
                row = (wintypes.DWORD * 6).from_buffer(buffer, 4 + index * 24)
                if row[0] == 2:
                    rows.append({"pid": row[5], "port": socket.ntohs(row[2] & 0xffff),
                                 "address": socket.inet_ntoa(int(row[1]).to_bytes(4, "little"))})
            return rows
        raise OSError("Список портов меняется; повторите запуск.")

    def matching(self, root):
        return [item for item in self.processes()
                if (script := streamlit_script(item["arguments"], item["cwd"])) is not None
                and same_path(script, root / "app.py")]

    def terminate_owned(self, identity):
        # Only a recorded child generation may be stopped, never a reused PID.
        handle = self.kernel.OpenProcess(0x0001 | 0x0400 | 0x0010, False, identity["pid"])
        if not handle:
            return
        try:
            if self.identity(identity["pid"]) == identity:
                if not self.kernel.TerminateProcess(handle, 1):
                    raise OSError("Не удалось остановить собственный дочерний процесс.")
        finally:
            self.kernel.CloseHandle(handle)


def belongs_to_child(inspector, pid, child_identity):
    """Account for the Windows venv redirector without accepting other servers."""
    visited = set()
    while pid and pid not in visited and len(visited) < 16:
        visited.add(pid)
        item = inspector.identity(pid)
        if item is None:
            return False
        if item == child_identity:
            return True
        if item["created"] < child_identity["created"]:
            return False
        pid = item["parent_pid"]
    return False


def healthy(port):
    try:
        with build_opener(ProxyHandler({})).open(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
            return response.status == 200 and response.read(32).strip() == b"ok"
    except Exception:
        return False


def existing_instance(inspector, root, timeout=0):
    deadline = time.monotonic() + timeout
    while True:
        processes = inspector.matching(root)
        ports = inspector.listeners()
        for process in processes:
            for listener in ports:
                if listener["pid"] == process["pid"] and listener["address"] in {"127.0.0.1", "0.0.0.0"} and healthy(listener["port"]):
                    # Re-read identity after HTTP to avoid accepting a recycled PID.
                    checked = inspector.identity(process["pid"])
                    if checked == process:
                        return {"pid": process["pid"], "port": listener["port"],
                                "url": f"http://127.0.0.1:{listener['port']}/"}
        if not processes:
            return None
        if time.monotonic() >= deadline:
            raise RuntimeError("Найден процесс этого симулятора, но его страница ещё не готова. Подождите и повторите запуск; второй экземпляр не создан.")
        time.sleep(0.5)


def free_port(start):
    for port in range(start, min(start + 100, 65536)):
        with socket.socket() as candidate:
            if os.name == "nt":
                candidate.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                candidate.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("Нет свободного локального порта. Закройте ненужный сервер или повторите позже; чужие процессы не остановлены.")


class ProjectMutex:
    def __init__(self, inspector, root):
        self.kernel = inspector.kernel
        self.kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        self.kernel.CreateMutexW.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
        name = "Local\\AdsorptionSimulator_" + hashlib.sha256(os.path.normcase(str(root)).encode("utf-8")).hexdigest()
        self.handle = self.kernel.CreateMutexW(None, False, name)
        if not self.handle:
            raise OSError("Не удалось создать защиту от повторного запуска.")
        self.owned = self.kernel.WaitForSingleObject(self.handle, 0) in (0, 128)

    def close(self):
        if self.owned:
            self.kernel.ReleaseMutex(self.handle)
        self.kernel.CloseHandle(self.handle)


def open_page(instance, no_browser):
    print(f"Симулятор: {instance['url']}", flush=True)
    if not no_browser and not webbrowser.open(instance["url"], new=2):
        print("Браузер не открылся автоматически. Откройте указанный адрес вручную.", flush=True)


class WindowsShortcut:
    """Use IShellLinkW, not the ANSI WScript.Shell shortcut interface."""

    @staticmethod
    def guid(value):
        return (ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID(value).bytes_le)

    @staticmethod
    def call(pointer, index, types, *args):
        table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        function = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *types)(table[index])
        result = function(pointer, *args)
        if result < 0:
            raise OSError(f"Windows не выполнила операцию с ярлыком (HRESULT {result & 0xffffffff:#x}).")

    def __enter__(self):
        self.ole = ctypes.OleDLL("ole32")
        self.ole.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        result = self.ole.CoInitializeEx(None, 2)
        if result < 0:
            raise OSError("Не удалось инициализировать создание ярлыка Windows.")
        self.link = ctypes.c_void_p()
        self.persist = ctypes.c_void_p()
        self.ole.CoCreateInstance.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
                                             ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        try:
            clsid = self.guid("00021401-0000-0000-C000-000000000046")
            iid = self.guid("000214F9-0000-0000-C000-000000000046")
            if self.ole.CoCreateInstance(ctypes.byref(clsid), None, 1, ctypes.byref(iid), ctypes.byref(self.link)) < 0:
                raise OSError("Windows не предоставила IShellLinkW.")
            persist_iid = self.guid("0000010B-0000-0000-C000-000000000046")
            self.call(self.link, 0, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
                      ctypes.byref(persist_iid), ctypes.byref(self.persist))
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for pointer in (self.persist, self.link):
            if pointer:
                self.call(pointer, 2, [])
        self.ole.CoUninitialize()

    def load(self, path):
        self.call(self.persist, 5, [wintypes.LPCWSTR, wintypes.DWORD], str(path), 0)

    def details(self):
        fields = {}
        for name, index in (("target", 3), ("cwd", 8), ("arguments", 10)):
            buffer = ctypes.create_unicode_buffer(32768)
            types = [wintypes.LPWSTR, ctypes.c_int]
            args = [buffer, len(buffer)]
            if index == 3:
                types += [ctypes.c_void_p, wintypes.DWORD]
                args += [None, 4]
            self.call(self.link, index, types, *args)
            fields[name] = buffer.value
        return fields

    def save(self, path, target, cwd):
        self.call(self.link, 20, [wintypes.LPCWSTR], str(target))
        self.call(self.link, 9, [wintypes.LPCWSTR], str(cwd))
        self.call(self.link, 17, [wintypes.LPCWSTR, ctypes.c_int], str(cwd / ".venv/Scripts/python.exe"), 0)
        self.call(self.persist, 6, [wintypes.LPCWSTR, wintypes.BOOL], str(path), True)


def create_shortcut(directory=None):
    if directory is None:
        desktop = ctypes.create_unicode_buffer(32768)
        shell = ctypes.WinDLL("shell32")
        shell.SHGetFolderPathW.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.HANDLE,
                                          wintypes.DWORD, wintypes.LPWSTR]
        if shell.SHGetFolderPathW(None, 0x10, None, 0, desktop) < 0:
            raise OSError("Папка рабочего стола недоступна.")
        directory = Path(desktop.value)
    directory = directory.resolve()
    if not directory.is_dir():
        raise OSError("Папка для ярлыка не существует.")
    target = ROOT / "Запустить_симулятор.bat"
    path = directory / "Запустить_симулятор.lnk"
    with WindowsShortcut() as shortcut:
        if path.exists():
            shortcut.load(path)
            if not same_path(shortcut.details()["target"], target):
                raise RuntimeError("Ярлык с таким именем ведёт к другой программе. Он не заменён; переименуйте его или выберите другое имя.")
        shortcut.save(path, target, ROOT)
    print(f"Ярлык создан: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-shortcut", action="store_true")
    parser.add_argument("--shortcut-directory", type=Path)
    parser.add_argument("--no-browser", action="store_true", help="For launcher smoke tests only")
    parser.add_argument("--probe", action="store_true", help="Read-only instance identity check")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise RuntimeError("Некорректный номер порта.")
    if not (ROOT / "app.py").is_file():
        raise RuntimeError("В папке launcher не найден app.py. Верните файл в корень проекта.")
    if args.create_shortcut:
        create_shortcut(args.shortcut_directory)
        return
    inspector = WindowsProcesses()
    if args.probe:
        print(json.dumps(existing_instance(inspector, ROOT), ensure_ascii=False), flush=True)
        return
    import importlib.util
    if importlib.util.find_spec("streamlit") is None:
        raise RuntimeError("В выбранном Python отсутствует Streamlit. Используйте рабочую .venv проекта; зависимости автоматически не устанавливаются.")
    mutex = ProjectMutex(inspector, ROOT)
    child = None
    child_identity = None
    owned_server = None
    try:
        instance = existing_instance(inspector, ROOT, timeout=30)
        if instance:
            print("Этот экземпляр уже работает. Открывается его страница; новый процесс не запускается.", flush=True)
            open_page(instance, args.no_browser)
            return
        if not mutex.owned:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                instance = existing_instance(inspector, ROOT, timeout=1)
                if instance:
                    open_page(instance, args.no_browser)
                    return
                time.sleep(0.5)
            raise RuntimeError("Другой launcher этой папки уже запускает программу. Проверьте его окно; второй экземпляр не создан.")
        port = free_port(args.port)
        if port != args.port:
            print(f"Порт {args.port} занят другим сервером. Используется свободный порт {port}; чужой процесс не остановлен.", flush=True)
        command = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
                   "--server.address=127.0.0.1", f"--server.port={port}", "--server.headless=true"]
        print(f"Python: {sys.executable}\nПапка проекта: {ROOT}\nЗапуск Streamlit...", flush=True)
        child = subprocess.Popen(command, cwd=ROOT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        child_identity = inspector.identity(child.pid)
        if child_identity is None:
            raise RuntimeError("Windows не разрешила проверить собственный процесс. Запуск остановлен; настройки безопасности не изменены.")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError(f"Streamlit завершился при запуске (код {child.returncode}). Причина указана выше.")
            if healthy(port):
                verified = existing_instance(inspector, ROOT)
                if verified and verified["port"] == port and belongs_to_child(inspector, verified["pid"], child_identity):
                    owned_server = inspector.identity(verified["pid"])
                    open_page(verified, args.no_browser)
                    break
            time.sleep(0.5)
        else:
            raise RuntimeError("Streamlit не открыл страницу за 60 секунд. Проверьте сообщения выше; расчёты и файлы проекта не изменены.")
        print("Не закрывайте это окно во время работы. Для остановки нажмите Ctrl+C в этом окне.", flush=True)
        # A finite polling wait leaves Windows console interrupts responsive.
        while child.poll() is None:
            time.sleep(0.5)
        if child.returncode != 0:
            raise RuntimeError(f"Streamlit завершился с ошибкой (код {child.returncode}). Проверьте сообщения выше.")
    except KeyboardInterrupt:
        print("\nОстановка только экземпляра, запущенного этим окном...", flush=True)
    finally:
        # Redirector/control events must not interrupt cleanup halfway through.
        previous_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        previous_break = signal.signal(signal.SIGBREAK, signal.SIG_IGN)
        owned_processes = []
        if child_identity is not None:
            owned_processes = [item for item in inspector.matching(ROOT)
                               if belongs_to_child(inspector, item["pid"], child_identity)]
        if child is not None:
            print("Завершение собственного сервера...", flush=True)
        if child is not None and child.poll() is None:
            inspector.kernel.GenerateConsoleCtrlEvent(1, child.pid)
            print("Сигнал остановки отправлен.", flush=True)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Сервер не ответил на сигнал; завершается только собственный процесс.", flush=True)
                child.terminate()
                child.wait(timeout=10)
        for item in owned_processes:
            inspector.terminate_owned(item)
        if owned_server is not None and owned_server not in owned_processes:
            inspector.terminate_owned(owned_server)
        mutex.close()
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGBREAK, previous_break)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if os.name == "nt":
        def interrupted(_signum, _frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGBREAK, interrupted)
    try:
        main()
    except Exception as error:
        print(f"\nОшибка запуска: {error}\nНе закрывайте окно, пока не прочитаете причину.", flush=True)
        sys.exit(1)
