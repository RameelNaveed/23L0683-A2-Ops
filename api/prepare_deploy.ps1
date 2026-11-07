# Copies the shared code and the champion model into the api folder so that
# "gcloud run deploy --source api" has everything it needs in one build folder.
# Run from the project root:  powershell -ExecutionPolicy Bypass -File api\prepare_deploy.ps1
Remove-Item -Recurse -Force api\visionguard -ErrorAction SilentlyContinue
Copy-Item -Recurse visionguard api\visionguard
Copy-Item models\champion.keras api\champion.keras -Force
Write-Host "api folder is ready to deploy:"
Get-ChildItem api
