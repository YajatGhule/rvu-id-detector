/* RVU ID Card Detector - Windows setup program.
 *
 * The project (a zip file) is glued to the end of this .exe, followed by a 16 byte trailer:
 * 8 bytes "RVUPAYLD" and the zip size as an 8 byte number. The program:
 *   1. copies the zip out of itself into the TEMP folder
 *   2. asks PowerShell to unzip it into %USERPROFILE%\rvu-id-detector
 *   3. runs installer\setup-local.ps1, which runs install-windows.ps1 and makes a Desktop shortcut
 *
 * Build on Linux: see installer/build.sh */
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>

static void wait_enter(void) {
    printf("\nPress Enter to close this window.");
    getchar();
}

static int fail(const char *msg) {
    printf("Setup failed: %s (error %lu)\n", msg, GetLastError());
    wait_enter();
    return 1;
}

/* double every ' so the path is safe inside a PowerShell '...' string */
static void ps_quote(const wchar_t *in, wchar_t *out, size_t cap) {
    size_t j = 0;
    for (size_t i = 0; in[i] && j + 2 < cap; i++) {
        if (in[i] == L'\'') out[j++] = L'\'';
        out[j++] = in[i];
    }
    out[j] = 0;
}

int main(void) {
    printf("RVU ID Card Detector - setup\n");
    printf("This copies the project to your user folder and installs Python and the packages.\n");
    printf("Internet is needed the first time. Nothing else on the computer is changed.\n\n");

    wchar_t self[MAX_PATH];
    if (!GetModuleFileNameW(NULL, self, MAX_PATH)) return fail("cannot find this program");

    HANDLE f = CreateFileW(self, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
    if (f == INVALID_HANDLE_VALUE) return fail("cannot open this program");

    LARGE_INTEGER total, pos;
    GetFileSizeEx(f, &total);
    unsigned char trailer[16];
    DWORD got;
    pos.QuadPart = total.QuadPart - 16;
    SetFilePointerEx(f, pos, NULL, FILE_BEGIN);
    if (!ReadFile(f, trailer, 16, &got, NULL) || got != 16 || memcmp(trailer, "RVUPAYLD", 8) != 0) {
        CloseHandle(f);
        return fail("the project files are missing from this setup program");
    }
    unsigned long long size;
    memcpy(&size, trailer + 8, 8);

    wchar_t temp[MAX_PATH], zip[MAX_PATH];
    GetTempPathW(MAX_PATH, temp);
    swprintf(zip, MAX_PATH, L"%lsrvu-id-detector.zip", temp);
    HANDLE out = CreateFileW(zip, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, 0, NULL);
    if (out == INVALID_HANDLE_VALUE) { CloseHandle(f); return fail("cannot write to the TEMP folder"); }

    printf("Step 1: copying the project files (%llu MB)\n", size / (1024 * 1024));
    pos.QuadPart = total.QuadPart - 16 - (long long)size;
    SetFilePointerEx(f, pos, NULL, FILE_BEGIN);
    static unsigned char buf[1 << 16];
    unsigned long long left = size;
    while (left > 0) {
        DWORD want = left > sizeof(buf) ? sizeof(buf) : (DWORD)left, wrote;
        if (!ReadFile(f, buf, want, &got, NULL) || got == 0) break;
        WriteFile(out, buf, got, &wrote, NULL);
        left -= got;
    }
    CloseHandle(out);
    CloseHandle(f);
    if (left != 0) return fail("the project files could not be read");

    wchar_t home[MAX_PATH], dest[MAX_PATH];
    if (!GetEnvironmentVariableW(L"USERPROFILE", home, MAX_PATH)) return fail("cannot find the user folder");
    swprintf(dest, MAX_PATH, L"%ls\\rvu-id-detector", home);

    wchar_t qzip[2 * MAX_PATH], qdest[2 * MAX_PATH];
    ps_quote(zip, qzip, 2 * MAX_PATH);
    ps_quote(dest, qdest, 2 * MAX_PATH);

    static wchar_t cmd[8192];
    swprintf(cmd, 8192,
        L"powershell.exe -NoProfile -ExecutionPolicy Bypass -Command \""
        L"Write-Host 'Step 2: unzipping to %ls'; "
        L"Expand-Archive -Force -LiteralPath '%ls' -DestinationPath '%ls'; "
        L"Write-Host 'Step 3: installing'; "
        L"& (Join-Path '%ls' 'installer\\setup-local.ps1'); exit $LASTEXITCODE\"",
        qdest, qzip, qdest, qdest);

    STARTUPINFOW si = {sizeof(si)};
    PROCESS_INFORMATION pi;
    if (!CreateProcessW(NULL, cmd, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi))
        return fail("PowerShell could not be started");
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    DeleteFileW(zip);

    printf(code == 0 ? "\nSetup finished. Open 'RVU ID Card Detector' on the Desktop.\n"
                     : "\nSetup did not finish. Read the message above.\n");
    wait_enter();
    return (int)code;
}
