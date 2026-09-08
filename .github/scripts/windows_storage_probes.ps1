# CI supervisor only. Administrator privileges prepare the sandbox; Python and
# all probe children run as a newly created, non-administrator local account.
param(
    [Parameter(Mandatory = $true)][string]$PythonPath,
    [ValidateSet('storage', 'appcontainer')][string]$Probe = 'storage'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$diagnostics = Join-Path $env:GITHUB_WORKSPACE 'windows-diagnostics'
New-Item -ItemType Directory -Force -Path $diagnostics | Out-Null
$logPrefix = if ($Probe -eq 'storage') { 'standard-user' } else { 'appcontainer' }
Start-Transcript -Path (Join-Path $diagnostics "$logPrefix-supervisor.log") | Out-Null
$user = $null
$process = $null
$started = $false
$password = $null
$taskName = $null
$comClsid = $null
$comAppid = $null
$setupError = $null
$exitCode = 1
$root = Join-Path $env:ProgramData ('LokiStorageProbes-' + [guid]::NewGuid().ToString('N'))

function Set-ProbeDirectoryAcl([string]$Path, [string]$UserSid, [string]$Rights) {
    # Only new disposable CI directories are changed, never user installations.
    $acl = [System.Security.AccessControl.DirectorySecurity]::new()
    $acl.SetAccessRuleProtection($true, $false)
    foreach ($entry in @(@('S-1-5-18', 'FullControl'),
                         @('S-1-5-32-544', 'FullControl'),
                         @($UserSid, $Rights))) {
        $sid = [System.Security.Principal.SecurityIdentifier]::new($entry[0])
        $rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
            $sid, $entry[1], 'ContainerInherit, ObjectInherit', 'None', 'Allow')
        $acl.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $Path -AclObject $acl
}

try {
    # Query only the selected interpreter, not an ambient Python after switching.
    $layout = & $PythonPath -I -c 'import json, os, sys; assert os.name == "nt"; print(json.dumps({"prefix": sys.base_prefix, "executable": os.path.relpath(sys.executable, sys.base_prefix)}))'
    if ($LASTEXITCODE -ne 0) { throw 'Cannot query the selected native interpreter' }
    $layout = $layout | ConvertFrom-Json
    if ([IO.Path]::IsPathRooted($layout.executable) -or $layout.executable.StartsWith('..')) {
        throw 'Selected executable is outside its interpreter prefix'
    }

    $name = 'loki_' + [guid]::NewGuid().ToString('N').Substring(0, 12)
    $random = [byte[]]::new(32)
    [Security.Cryptography.RandomNumberGenerator]::Fill($random)
    $plain = 'Aa1!' + [Convert]::ToBase64String($random)
    # Never pass the password via argv, environment, a file, or workflow outputs.
    $password = ConvertTo-SecureString $plain -AsPlainText -Force
    $plain = $null
    $user = New-LocalUser -Name $name -Password $password -Description 'Disposable Loki CI probes'
    $sid = $user.SID.Value
    $users = Get-LocalGroup -SID 'S-1-5-32-545'
    if ($sid -notin @(Get-LocalGroupMember -Group $users | ForEach-Object { $_.SID.Value })) {
        Add-LocalGroupMember -Group $users -Member $user
    }
    Write-Host "Standard-user account: $name; expected SID: $sid"

    New-Item -ItemType Directory -Path $root | Out-Null
    # The AppContainer broker must configure ACLs on these fresh copies as a
    # standard user. Its contained child receives only separately granted RX.
    $stageRights = if ($Probe -eq 'appcontainer') { 'FullControl' } else { 'ReadAndExecute' }
    Set-ProbeDirectoryAcl $root $sid $stageRights
    $runtime = Join-Path $root 'runtime'
    # Copying avoids granting this user access to the administrator's toolcache
    # or MSYS2 installation. Reset only the copies to the CI stage's ACL.
    Copy-Item -LiteralPath $layout.prefix -Destination $runtime -Recurse
    $script = Join-Path $root 'test_windows_primitives.py'
    Copy-Item -LiteralPath (Join-Path $env:GITHUB_WORKSPACE 'tests/test_windows_primitives.py') -Destination $script
    foreach ($copy in @($runtime, $script)) {
        & "$env:SystemRoot/System32/icacls.exe" $copy /reset /T /Q
        if ($LASTEXITCODE -ne 0) { throw 'Cannot set stage ACLs on probe copies' }
    }
    if ($Probe -eq 'appcontainer') {
        $script = Join-Path $root 'test_windows_appcontainers.py'
        foreach ($probeFile in @('test_windows_appcontainers.py', 'windows_escapes.py')) {
            $copy = Join-Path $root $probeFile
            Copy-Item -LiteralPath (Join-Path $env:GITHUB_WORKSPACE "tests/$probeFile") -Destination $copy
            & "$env:SystemRoot/System32/icacls.exe" $copy /reset /Q
            if ($LASTEXITCODE -ne 0) { throw 'Cannot set ACL on AppContainer probe copy' }
        }
    }
    $work = Join-Path $root 'work'
    New-Item -ItemType Directory -Path $work | Out-Null
    Set-ProbeDirectoryAcl $work $sid 'FullControl'
    $temporary = Join-Path $work 'tmp'
    New-Item -ItemType Directory -Path $temporary | Out-Null
    $executable = Join-Path $runtime $layout.executable

    if ($Probe -eq 'appcontainer') {
        # Decoupled, best-effort task fixture: every registration path was
        # denied on this image even with SeBatchLogonRight granted (native
        # runs 1670996, cd5802c, 8a7f800); the cause is unverified. A denial
        # must not abort the containment probes: it leaves no fixture file,
        # and the broker records the route as setup-blocked instead.
        $taskName = 'LokiProbe-' + [guid]::NewGuid().ToString('N').Substring(0, 12)
        $request = Join-Path $root 'task-request.json'
        $witnessArguments = '-I -u "{0}" --task-witness "{1}"' -f $script, $request
        $template = @'
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Principals>
    <Principal id="Author">
      <UserId>{0}</UserId>
      <LogonType>S4U</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <ExecutionTimeLimit>PT5M</ExecutionTimeLimit>
  </Settings>
  <Actions>
    <Exec>
      <Command>{1}</Command>
      <Arguments>{2}</Arguments>
    </Exec>
  </Actions>
</Task>
'@
        $definition = Join-Path $root 'task-definition.xml'
        # Parenthesized: bare -f in argument mode binds only the first item,
        # which failed the three-placeholder template (native run 098ddb7).
        $taskXml = $template -f $sid, $executable, $witnessArguments
        [IO.File]::WriteAllText($definition, $taskXml, [Text.Encoding]::Unicode)
        $registration = & "$env:SystemRoot/System32/schtasks.exe" /Create /TN $taskName /XML $definition /F 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Scheduled-task fixture blocked: $registration"
        }
        else {
            try {
                # The admin-created default descriptor may omit the run-as
                # user; grant this disposable account read-and-execute.
                $task = Get-ScheduledTask -TaskName $taskName
                $task.SecurityDescriptorSddl = 'D:P(A;;FA;;;BA)(A;;FA;;;SY)(A;;FRFX;;;' + $sid + ')'
                $task | Set-ScheduledTask | Out-Null
            }
            catch { Write-Warning "Task security descriptor update failed: $_" }
            @{ task = $taskName; request = $request } | ConvertTo-Json |
                Set-Content -LiteralPath (Join-Path $root 'task-fixture.json')
        }
    }

    if ($Probe -eq 'appcontainer') {
        # Machine-wide local COM server (arm 2): the administrator registers
        # one disposable machine class whose fixed bootstrap reads its witness
        # target from the broker-owned request file. LaunchPermission grants
        # only the temp user, SYSTEM and Administrators -- never ALL
        # APPLICATION PACKAGES, which is the denial the worker attempt is
        # meant to exercise. COM requires LaunchPermission as REG_BINARY
        # (a self-relative SECURITY_DESCRIPTOR); an SDDL string is not
        # accepted. Best effort: a failure prints and leaves no fixture
        # file; probes still run.
        $comClsid = '{' + [guid]::NewGuid().ToString() + '}'
        $comAppid = '{' + [guid]::NewGuid().ToString() + '}'
        $comRequest = Join-Path $root 'com-request.json'
        $comCommand = '"{0}" -I -u "{1}" --com-witness "{2}"' -f $executable, $script, $comRequest
        $comClsidPath = 'HKLM:\SOFTWARE\Classes\CLSID\' + $comClsid
        $comAppidPath = 'HKLM:\SOFTWARE\Classes\AppID\' + $comAppid
        try {
            if (-not ('Loki.ComSd' -as [type])) {
                Add-Type -Namespace Loki -Name ComSd -MemberDefinition @'
[DllImport("advapi32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
public static extern bool ConvertStringSecurityDescriptorToSecurityDescriptor(
    string sddl, uint revision, out IntPtr descriptor, out uint length);
[DllImport("kernel32.dll", SetLastError=true)]
public static extern IntPtr LocalFree(IntPtr memory);
'@
            }
            $descriptor = [IntPtr]::Zero
            $descriptorSize = [uint32]0
            $launchSddl = 'O:SYD:P(A;;GA;;;' + $sid + ')(A;;GA;;;SY)(A;;GA;;;BA)'
            $converted = [Loki.ComSd]::ConvertStringSecurityDescriptorToSecurityDescriptor(
                $launchSddl, 1, [ref]$descriptor, [ref]$descriptorSize)
            if (-not $converted) { throw 'Cannot build COM launch permission descriptor' }
            $launchBytes = [byte[]]::new([int]$descriptorSize)
            [Runtime.InteropServices.Marshal]::Copy(
                $descriptor, $launchBytes, 0, [int]$descriptorSize)
            [void][Loki.ComSd]::LocalFree($descriptor)
            New-Item -Path ($comClsidPath + '\LocalServer32') -Force | Out-Null
            Set-ItemProperty -Path ($comClsidPath + '\LocalServer32') -Name '(default)' -Value $comCommand
            Set-ItemProperty -Path $comClsidPath -Name 'AppID' -Value $comAppid
            New-Item -Path $comAppidPath -Force | Out-Null
            Set-ItemProperty -Path $comAppidPath -Name 'LaunchPermission' `
                -Type Binary -Value $launchBytes
            @{ clsid = $comClsid; request = $comRequest } | ConvertTo-Json |
                Set-Content -LiteralPath (Join-Path $root 'com-fixture.json')
        }
        catch { Write-Host "COM fixture blocked: $_" }
    }

    $start = [Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $executable
    $start.WorkingDirectory = $work
    $start.UseShellExecute = $false
    $start.UserName = $name
    $start.Domain = $env:COMPUTERNAME
    $start.Password = $password
    $start.LoadUserProfile = $true
    $start.RedirectStandardInput = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    foreach ($argument in @('-I', '-u', $script, '--standard-user', $sid)) {
        $start.ArgumentList.Add($argument)
    }
    # Do not give the child the runner's credentials, Python configuration,
    # user-home paths, or arbitrary inherited PATH entries.
    $start.Environment.Clear()
    $start.Environment['SystemRoot'] = $env:SystemRoot
    $start.Environment['WINDIR'] = $env:SystemRoot
    $start.Environment['COMSPEC'] = Join-Path $env:SystemRoot 'System32/cmd.exe'
    $start.Environment['PATH'] = (Split-Path $executable) + ';' + (Join-Path $env:SystemRoot 'System32')
    $start.Environment['TEMP'] = $temporary
    $start.Environment['TMP'] = $temporary

    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $start
    if (-not $process.Start()) { throw 'Standard-user Python did not start' }
    $started = $true
    $process.StandardInput.Close()
    # Drain both streams concurrently, so verbose failures cannot fill a pipe.
    $stdout = $process.StandardOutput.ReadToEndAsync()
    $stderr = $process.StandardError.ReadToEndAsync()
    $timedOut = -not $process.WaitForExit(180000)
    if ($timedOut) {
        $process.Kill($true)
        if (-not $process.WaitForExit(10000)) { throw 'Timed-out probe tree did not exit' }
    }
    foreach ($entry in @(@($stdout, "$logPrefix-stdout.log"),
                         @($stderr, "$logPrefix-stderr.log"))) {
        if (-not $entry[0].Wait(10000)) { throw 'Probe output pipe did not close' }
        $text = $entry[0].GetAwaiter().GetResult()
        [IO.File]::WriteAllText((Join-Path $diagnostics $entry[1]), $text)
        Write-Host $text
    }
    $exitCode = $process.ExitCode
    Write-Host "Standard-user probe exit: $exitCode; timed out: $timedOut"
    if ($timedOut) { throw 'Standard-user probes exceeded 180 seconds' }
}
catch {
    # Preserve the original failure; a finally throw would otherwise mask it
    # (demonstrated at 77615b0, where the cleanup error hid the real one).
    $setupError = $_
}
finally {
    # Attempt every cleanup even if an earlier one fails. Hard job termination
    # may bypass finally; the hosted VM is disposable and no account is reused.
    $cleanupFailed = $false
    if ($null -ne $process) {
        try {
            if ($started -and -not $process.HasExited) {
                $process.Kill($true)
                if (-not $process.WaitForExit(10000)) { throw 'Probe process did not exit' }
            }
        }
        catch { Write-Warning $_; $cleanupFailed = $true }
        finally { $process.Dispose() }
    }
    if ($taskName) {
        # A setup failure before registration leaves no task; that is quiet,
        # not a cleanup failure.
        $registered = Get-ScheduledTask -TaskName $taskName -ErrorAction Ignore -WarningAction Ignore
        if ($registered) {
            try {
                Stop-ScheduledTask -TaskName $taskName -ErrorAction Ignore -WarningAction Ignore
                Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
            }
            catch { Write-Warning $_; $cleanupFailed = $true }
        }
    }
    if ($comClsid -or $comAppid) {
        # Quiet removal for keys a blocked registration never wrote; real
        # failures are warnings.
        try {
            if ($comClsid -and (Test-Path ('HKLM:\SOFTWARE\Classes\CLSID\' + $comClsid))) {
                Remove-Item -Path ('HKLM:\SOFTWARE\Classes\CLSID\' + $comClsid) -Recurse -Force
            }
            if ($comAppid -and (Test-Path ('HKLM:\SOFTWARE\Classes\AppID\' + $comAppid))) {
                Remove-Item -Path ('HKLM:\SOFTWARE\Classes\AppID\' + $comAppid) -Recurse -Force
            }
        }
        catch { Write-Warning $_; $cleanupFailed = $true }
    }
    if ($null -ne $user) {
        if ($Probe -eq 'appcontainer') {
            # Broker-created witnesses may be outside the root's process tree.
            # Filter by this fresh, unique account; never kill by global image
            # name. This administrative cleanup is not containment evidence.
            try {
                & "$env:SystemRoot/System32/taskkill.exe" /F /FI "USERNAME eq $env:COMPUTERNAME\$($user.Name)"
                if ($LASTEXITCODE -ne 0) { throw 'Account-scoped process cleanup failed' }
            }
            catch { Write-Warning $_; $cleanupFailed = $true }
        }
        try {
            Get-CimInstance Win32_UserProfile | Where-Object SID -EQ $user.SID.Value | Remove-CimInstance
        }
        catch { Write-Warning $_; $cleanupFailed = $true }
        try { Remove-LocalUser -SID $user.SID }
        catch { Write-Warning $_; $cleanupFailed = $true }
    }
    try {
        if (Test-Path -LiteralPath $root) { Remove-Item -LiteralPath $root -Recurse -Force }
    }
    catch { Write-Warning $_; $cleanupFailed = $true }
    if ($null -ne $password) { $password.Dispose() }
    Stop-Transcript | Out-Null
}
# Rethrow the original failure after cleanup; only report cleanup problems
# when nothing earlier failed, so the first error is always the visible one.
if ($null -ne $setupError) { throw $setupError }
if ($cleanupFailed) { throw 'Standard-user supervisor cleanup failed; see transcript' }
exit $exitCode
